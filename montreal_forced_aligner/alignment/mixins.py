"""Class definitions for alignment mixins"""
from __future__ import annotations

import csv
import datetime
import logging
import multiprocessing as mp
import os
import time
from abc import abstractmethod
from queue import Empty
from typing import TYPE_CHECKING, Dict, List

import tqdm

from montreal_forced_aligner.alignment.multiprocessing import (
    AlignArguments,
    AlignFunction,
    CompileInformationArguments,
    CompileTrainGraphsArguments,
    CompileTrainGraphsFunction,
    PhoneConfidenceArguments,
    PhoneConfidenceFunction,
    compile_information_func,
)
from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.db import (
    CorpusWorkflow,
    File,
    Job,
    PhoneInterval,
    Speaker,
    Utterance,
    bulk_update,
)
from montreal_forced_aligner.dictionary.mixins import DictionaryMixin
from montreal_forced_aligner.exceptions import NoAlignmentsError
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.utils import KaldiProcessWorker, Stopped, run_mp, run_non_mp

if TYPE_CHECKING:
    from montreal_forced_aligner.abc import MetaDict


logger = logging.getLogger("mfa")


class AlignMixin(DictionaryMixin):
    """
    Configuration object for alignment

    Parameters
    ----------
    transition_scale : float
        Transition scale, defaults to 1.0
    acoustic_scale : float
        Acoustic scale, defaults to 0.1
    self_loop_scale : float
        Self-loop scale, defaults to 0.1
    boost_silence : float
        Factor to boost silence probabilities, 1.0 is no boost or reduction
    beam : int
        Size of the beam to use in decoding, defaults to 10
    retry_beam : int
        Size of the beam to use in decoding if it fails with the initial beam width, defaults to 40


    See Also
    --------
    :class:`~montreal_forced_aligner.dictionary.mixins.DictionaryMixin`
        For dictionary parsing parameters

    Attributes
    ----------
    jobs: list[:class:`~montreal_forced_aligner.corpus.multiprocessing.Job`]
        Jobs to process
    """

    logger: logging.Logger
    jobs: List[Job]

    def __init__(
        self,
        transition_scale: float = 1.0,
        acoustic_scale: float = 0.1,
        self_loop_scale: float = 0.1,
        boost_silence: float = 1.0,
        beam: int = 10,
        retry_beam: int = 40,
        fine_tune: bool = False,
        phone_confidence: bool = False,
        use_phone_model: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transition_scale = transition_scale
        self.acoustic_scale = acoustic_scale
        self.self_loop_scale = self_loop_scale
        self.boost_silence = boost_silence
        self.beam = beam
        self.retry_beam = retry_beam
        self.fine_tune = fine_tune
        self.phone_confidence = phone_confidence
        self.use_phone_model = use_phone_model
        if self.retry_beam <= self.beam:
            self.retry_beam = self.beam * 4
        self.unaligned_files = set()

    @property
    def tree_path(self) -> str:
        """Path to tree file"""
        return os.path.join(self.working_directory, "tree")

    @property
    @abstractmethod
    def data_directory(self) -> str:
        """Corpus data directory"""
        ...

    @abstractmethod
    def construct_feature_proc_strings(self) -> List[Dict[str, str]]:
        """Generate feature strings"""
        ...

    def compile_train_graphs_arguments(self) -> List[CompileTrainGraphsArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.CompileTrainGraphsFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.CompileTrainGraphsArguments`]
            Arguments for processing
        """
        args = []
        model_path = self.model_path
        if not os.path.exists(model_path):
            model_path = self.alignment_model_path
        for j in self.jobs:
            args.append(
                CompileTrainGraphsArguments(
                    j.id,
                    getattr(self, "db_string", ""),
                    os.path.join(self.working_log_directory, f"compile_train_graphs.{j.id}.log"),
                    os.path.join(self.working_directory, "tree"),
                    model_path,
                    getattr(self, "use_g2p", False),
                )
            )
        return args

    def align_arguments(self) -> List[AlignArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.AlignFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.AlignArguments`]
            Arguments for processing
        """
        args = []
        iteration = getattr(self, "iteration", None)
        for j in self.jobs:
            if iteration is not None:
                log_path = os.path.join(
                    self.working_log_directory, f"align.{iteration}.{j.id}.log"
                )
            else:
                log_path = os.path.join(self.working_log_directory, f"align.{j.id}.log")
            if getattr(self, "uses_speaker_adaptation", False):
                log_path = log_path.replace(".log", ".fmllr.log")
            args.append(
                AlignArguments(
                    j.id,
                    getattr(self, "db_string", ""),
                    log_path,
                    self.alignment_model_path,
                    self.decode_options
                    if self.phone_confidence
                    and getattr(self, "uses_speaker_adaptation", False)
                    and hasattr(self, "decode_options")
                    else self.align_options,
                    self.feature_options,
                    self.phone_confidence,
                )
            )
        return args

    def phone_confidence_arguments(self) -> List[PhoneConfidenceArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.PhoneConfidenceFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.PhoneConfidenceArguments`]
            Arguments for processing
        """
        args = []
        for j in self.jobs:
            log_path = os.path.join(self.working_log_directory, f"phone_confidence.{j.id}.log")

            feat_strings = {}
            for d in j.dictionaries:
                feat_strings[d.id] = j.construct_feature_proc_string(
                    self.working_directory,
                    d.id,
                    self.feature_options["uses_splices"],
                    self.feature_options["splice_left_context"],
                    self.feature_options["splice_right_context"],
                    self.feature_options["uses_speaker_adaptation"],
                )
            args.append(
                PhoneConfidenceArguments(
                    j.id,
                    getattr(self, "db_string", ""),
                    log_path,
                    self.model_path,
                    self.phone_pdf_counts_path,
                    feat_strings,
                )
            )
        return args

    def compile_information_arguments(self) -> List[CompileInformationArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.alignment.multiprocessing.compile_information_func`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.CompileInformationArguments`]
            Arguments for processing
        """
        args = []
        iteration = getattr(self, "iteration", None)
        for j in self.jobs:
            if iteration is not None:
                log_path = os.path.join(
                    self.working_log_directory, f"align.{iteration}.{j.id}.log"
                )
            else:
                log_path = os.path.join(self.working_log_directory, f"align.{j.id}.log")
            args.append(
                CompileInformationArguments(
                    j.id,
                    getattr(self, "db_string", ""),
                    os.path.join(self.working_log_directory, f"compile_information.{j.id}.log"),
                    log_path,
                )
            )
        return args

    @property
    def align_options(self) -> MetaDict:
        """Options for use in aligning"""

        return {
            "transition_scale": self.transition_scale,
            "acoustic_scale": self.acoustic_scale,
            "self_loop_scale": self.self_loop_scale,
            "beam": self.beam,
            "retry_beam": self.retry_beam,
            "boost_silence": self.boost_silence,
            "optional_silence_csl": self.optional_silence_csl,
        }

    def alignment_configuration(self) -> MetaDict:
        """Configuration parameters"""
        return {
            "transition_scale": self.transition_scale,
            "acoustic_scale": self.acoustic_scale,
            "self_loop_scale": self.self_loop_scale,
            "boost_silence": self.boost_silence,
            "beam": self.beam,
            "retry_beam": self.retry_beam,
        }

    @property
    def num_current_utterances(self) -> int:
        """Number of current utterances"""
        return getattr(self, "num_utterances", 0)

    def compile_train_graphs(self) -> None:
        """
        Multiprocessing function that compiles training graphs for utterances.

        See Also
        --------
        :class:`~montreal_forced_aligner.alignment.multiprocessing.CompileTrainGraphsFunction`
            Multiprocessing helper function for each job
        :meth:`.AlignMixin.compile_train_graphs_arguments`
            Job method for generating arguments for the helper function
        :kaldi_steps:`align_si`
            Reference Kaldi script
        :kaldi_steps:`align_fmllr`
            Reference Kaldi script
        """
        begin = time.time()
        log_directory = self.working_log_directory
        os.makedirs(log_directory, exist_ok=True)
        logger.info("Compiling training graphs...")
        error_sum = 0
        arguments = self.compile_train_graphs_arguments()
        with tqdm.tqdm(total=self.num_current_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
            if GLOBAL_CONFIG.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(arguments):
                    function = CompileTrainGraphsFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if isinstance(result, Exception):
                            error_dict[getattr(result, "job_name", 0)] = result
                            continue
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    done, errors = result
                    pbar.update(done + errors)
                    error_sum += errors
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                logger.debug("Not using multiprocessing...")
                for args in arguments:
                    function = CompileTrainGraphsFunction(args)
                    for done, errors in function.run():
                        pbar.update(done + errors)
                        error_sum += errors
        if error_sum:
            logger.warning(f"Compilation of training graphs failed for {error_sum} utterances.")
        logger.debug(f"Compiling training graphs took {time.time() - begin:.3f} seconds")

    def get_phone_confidences(self):
        if not os.path.exists(self.phone_pdf_counts_path):
            logger.warning("Cannot calculate phone confidences with the current model.")
            return
        logger.info("Calculating phone confidences...")
        begin = time.time()

        with self.session() as session:
            with tqdm.tqdm(total=self.num_current_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
                arguments = self.phone_confidence_arguments()
                interval_update_mappings = []
                if GLOBAL_CONFIG.use_mp:
                    error_dict = {}
                    return_queue = mp.Queue()
                    stopped = Stopped()
                    procs = []
                    for i, args in enumerate(arguments):
                        function = PhoneConfidenceFunction(args)
                        p = KaldiProcessWorker(i, return_queue, function, stopped)
                        procs.append(p)
                        p.start()
                    while True:
                        try:
                            result = return_queue.get(timeout=1)
                            if isinstance(result, Exception):
                                error_dict[getattr(result, "job_name", 0)] = result
                                continue
                            if stopped.stop_check():
                                continue
                        except Empty:
                            for proc in procs:
                                if not proc.finished.stop_check():
                                    break
                            else:
                                break
                            continue
                        interval_update_mappings.extend(result)
                        pbar.update(1)
                    for p in procs:
                        p.join()

                    if error_dict:
                        for v in error_dict.values():
                            raise v

                else:
                    logger.debug("Not using multiprocessing...")
                    for args in arguments:
                        function = PhoneConfidenceFunction(args)
                        for result in function.run():
                            interval_update_mappings.extend(result)
                            pbar.update(1)
                bulk_update(session, PhoneInterval, interval_update_mappings)
                session.commit()
            logger.debug(f"Calculating phone confidences took {time.time() - begin:.3f} seconds")

    def align_utterances(self, training=False) -> None:
        """
        Multiprocessing function that aligns based on the current model.

        See Also
        --------
        :class:`~montreal_forced_aligner.alignment.multiprocessing.AlignFunction`
            Multiprocessing helper function for each job
        :meth:`.AlignMixin.align_arguments`
            Job method for generating arguments for the helper function
        :kaldi_steps:`align_si`
            Reference Kaldi script
        :kaldi_steps:`align_fmllr`
            Reference Kaldi script
        """
        begin = time.time()
        logger.info("Generating alignments...")
        with tqdm.tqdm(
            total=self.num_current_utterances, disable=GLOBAL_CONFIG.quiet
        ) as pbar, self.session() as session:
            if not training:
                utterances = session.query(Utterance)
                if hasattr(self, "subset"):
                    utterances = utterances.filter(Utterance.in_subset == True)  # noqa
                utterances.update({"alignment_log_likelihood": None})
                session.commit()
            log_like_sum = 0
            log_like_count = 0
            update_mappings = []
            if GLOBAL_CONFIG.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(self.align_arguments()):
                    function = AlignFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if isinstance(result, Exception):
                            error_dict[getattr(result, "job_name", 0)] = result
                            continue
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    if not training:
                        utterance, log_likelihood = result
                        log_like_sum += log_likelihood
                        log_like_count += 1
                        update_mappings.append(
                            {"id": utterance, "alignment_log_likelihood": log_likelihood}
                        )
                    pbar.update(1)
                for p in procs:
                    p.join()

                if not training and len(update_mappings) == 0:
                    raise NoAlignmentsError(
                        self.num_current_utterances, self.beam, self.retry_beam
                    )
                if error_dict:
                    for v in error_dict.values():
                        raise v

            else:
                logger.debug("Not using multiprocessing...")
                for args in self.align_arguments():
                    function = AlignFunction(args)
                    for utterance, log_likelihood in function.run():
                        if not training:
                            log_like_sum += log_likelihood
                            log_like_count += 1
                            update_mappings.append(
                                {"id": utterance, "alignment_log_likelihood": log_likelihood}
                            )
                        pbar.update(1)
                if not training and len(update_mappings) == 0:
                    raise NoAlignmentsError(
                        self.num_current_utterances, self.beam, self.retry_beam
                    )
            if not training:
                bulk_update(session, Utterance, update_mappings)
                session.query(Utterance).filter(
                    Utterance.alignment_log_likelihood != None  # noqa
                ).update(
                    {
                        Utterance.alignment_log_likelihood: Utterance.alignment_log_likelihood
                        / Utterance.num_frames
                    },
                    synchronize_session="fetch",
                )
                if not training:
                    if not getattr(self, "uses_speaker_adaptation", False):
                        workflow = (
                            session.query(CorpusWorkflow)
                            .filter(CorpusWorkflow.current == True)  # noqa
                            .first()
                        )
                        workflow.time_stamp = datetime.datetime.now()
                        workflow.score = log_like_sum / log_like_count
                session.commit()
            logger.debug(f"Alignment round took {time.time() - begin:.3f} seconds")

    def compile_information(self) -> None:
        """
        Compiles information about alignment, namely what the overall log-likelihood was
        and how many files were unaligned.

        See Also
        --------
        :func:`~montreal_forced_aligner.alignment.multiprocessing.compile_information_func`
            Multiprocessing helper function for each job
        :meth:`.AlignMixin.compile_information_arguments`
            Job method for generating arguments for the helper function
        """
        compile_info_begin = time.time()

        jobs = self.compile_information_arguments()

        if GLOBAL_CONFIG.use_mp:
            alignment_info = run_mp(
                compile_information_func, jobs, self.working_log_directory, True
            )
        else:
            alignment_info = run_non_mp(
                compile_information_func, jobs, self.working_log_directory, True
            )

        avg_like_sum = 0
        avg_like_frames = 0
        average_logdet_sum = 0
        average_logdet_frames = 0
        beam_too_narrow_count = 0
        too_short_count = 0
        for data in alignment_info.values():
            beam_too_narrow_count += len(data["unaligned"])
            too_short_count += len(data["too_short"])
            avg_like_frames += data["total_frames"]
            avg_like_sum += data["log_like"] * data["total_frames"]
            if "logdet_frames" in data:
                average_logdet_frames += data["logdet_frames"]
                average_logdet_sum += data["logdet"] * data["logdet_frames"]

        if hasattr(self, "db_engine"):
            csv_path = os.path.join(self.working_directory, "alignment_log_likelihood.csv")
            with mfa_open(csv_path, "w") as f, self.session() as session:
                writer = csv.writer(f)
                writer.writerow(["file", "begin", "end", "speaker", "loglikelihood"])
                utterances = (
                    session.query(
                        File.name,
                        Utterance.begin,
                        Utterance.end,
                        Speaker.name,
                        Utterance.alignment_log_likelihood,
                    )
                    .join(Utterance.file)
                    .join(Utterance.speaker)
                    .filter(Utterance.alignment_log_likelihood != None)  # noqa
                )
                if hasattr(self, "subset"):
                    utterances = utterances.filter(Utterance.in_subset == True)  # noqa
                for file_name, begin, end, speaker_name, alignment_log_likelihood in utterances:
                    writer.writerow(
                        [file_name, begin, end, speaker_name, alignment_log_likelihood]
                    )

        if not avg_like_frames:
            logger.warning(
                "No files were aligned, this likely indicates serious problems with the aligner."
            )
        else:
            if too_short_count:
                logger.debug(
                    f"There were {too_short_count} utterances that were too short to be aligned."
                )
            if beam_too_narrow_count:
                logger.debug(
                    f"There were {beam_too_narrow_count} utterances that could not be aligned with "
                    f"the current beam settings."
                )
            average_log_like = avg_like_sum / avg_like_frames
            if average_logdet_sum:
                average_log_like += average_logdet_sum / average_logdet_frames
            logger.debug(f"Average per frame likelihood for alignment: {average_log_like}")
        logger.debug(f"Compiling information took {time.time() - compile_info_begin:.3f} seconds")

    @property
    @abstractmethod
    def working_directory(self) -> str:
        """Working directory"""
        ...

    @property
    @abstractmethod
    def working_log_directory(self) -> str:
        """Working log directory"""
        ...

    @property
    def model_path(self) -> str:
        """Acoustic model file path"""
        return os.path.join(self.working_directory, "final.mdl")

    @property
    def phone_pdf_counts_path(self) -> str:
        """Acoustic model file path"""
        return os.path.join(self.working_directory, "phone_pdf.counts")

    @property
    def alignment_model_path(self) -> str:
        """Acoustic model file path for speaker-independent alignment"""
        path = os.path.join(self.working_directory, "final.alimdl")
        if os.path.exists(path) and not getattr(self, "uses_speaker_adaptation", False):
            return path
        return self.model_path
