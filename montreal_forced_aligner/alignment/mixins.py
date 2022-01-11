"""Class definitions for alignment mixins"""
from __future__ import annotations

import csv
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
    compile_information_func,
)
from montreal_forced_aligner.dictionary.mixins import DictionaryMixin
from montreal_forced_aligner.utils import KaldiProcessWorker, Stopped, run_mp, run_non_mp

if TYPE_CHECKING:
    from montreal_forced_aligner.abc import MetaDict
    from montreal_forced_aligner.corpus.multiprocessing import Job


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
    logger: logging.Logger
        Eventual top-level worker logger
    jobs: list[Job]
        Jobs to process
    use_mp: bool
        Flag for using multiprocessing
    """

    logger: logging.Logger
    jobs: List[Job]
    use_mp: bool

    def __init__(
        self,
        transition_scale: float = 1.0,
        acoustic_scale: float = 0.1,
        self_loop_scale: float = 0.1,
        boost_silence: float = 1.0,
        beam: int = 10,
        retry_beam: int = 40,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transition_scale = transition_scale
        self.acoustic_scale = acoustic_scale
        self.self_loop_scale = self_loop_scale
        self.boost_silence = boost_silence
        self.beam = beam
        self.retry_beam = retry_beam
        if self.retry_beam <= self.beam:
            self.retry_beam = self.beam * 4
        self.unaligned_files = set()

    @property
    def tree_path(self):
        """Path to tree file"""
        return os.path.join(self.working_directory, "tree")

    @property
    @abstractmethod
    def data_directory(self):
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
        for j in self.jobs:
            lexicon_fst_paths = {
                dictionary.name: dictionary.lexicon_fst_path
                for dictionary in j.current_dictionaries
            }
            model_path = self.model_path
            if not os.path.exists(model_path):
                model_path = self.alignment_model_path
            args.append(
                CompileTrainGraphsArguments(
                    os.path.join(self.working_log_directory, f"compile_train_graphs.{j.name}.log"),
                    j.current_dictionary_names,
                    os.path.join(self.working_directory, "tree"),
                    model_path,
                    j.construct_path_dictionary(self.data_directory, "text", "int.scp"),
                    self.disambiguation_symbols_int_path,
                    lexicon_fst_paths,
                    j.construct_path_dictionary(self.working_directory, "fsts", "scp"),
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
        feat_strings = self.construct_feature_proc_strings()
        iteration = getattr(self, "iteration", None)
        for j in self.jobs:
            if iteration is not None:
                log_path = os.path.join(
                    self.working_log_directory, f"align.{iteration}.{j.name}.log"
                )
            else:
                log_path = os.path.join(self.working_log_directory, f"align.{j.name}.log")
            args.append(
                AlignArguments(
                    log_path,
                    j.current_dictionary_names,
                    j.construct_path_dictionary(self.working_directory, "fsts", "scp"),
                    feat_strings[j.name],
                    self.alignment_model_path,
                    j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                    self.align_options,
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
                    self.working_log_directory, f"align.{iteration}.{j.name}.log"
                )
            else:
                log_path = os.path.join(self.working_log_directory, f"align.{j.name}.log")
            args.append(CompileInformationArguments(log_path))
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
        self.logger.info("Compiling training graphs...")
        error_sum = 0
        arguments = self.compile_train_graphs_arguments()
        with tqdm.tqdm(total=self.num_utterances) as pbar:
            if self.use_mp:
                manager = mp.Manager()
                error_dict = manager.dict()
                return_queue = manager.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(arguments):
                    function = CompileTrainGraphsFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, error_dict, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        done, errors = return_queue.get(timeout=1)
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    pbar.update(done + errors)
                    error_sum += errors
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                self.logger.debug("Not using multiprocessing...")
                for args in arguments:
                    function = CompileTrainGraphsFunction(args)
                    for done, errors in function.run():
                        pbar.update(done + errors)
                        error_sum += errors
        if error_sum:
            self.logger.warning(
                f"Compilation of training graphs failed for {error_sum} utterances."
            )
        self.logger.debug(f"Compiling training graphs took {time.time() - begin}")

    def align_utterances(self) -> None:
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
        self.unaligned_files = set()
        self.logger.info("Generating alignments...")
        with tqdm.tqdm(total=self.num_utterances) as pbar:
            if self.use_mp:
                manager = mp.Manager()
                error_dict = manager.dict()
                return_queue = manager.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(self.align_arguments()):
                    function = AlignFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, error_dict, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        utterance, log_likelihood = return_queue.get(timeout=1)
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    if hasattr(self, "utterances"):
                        if hasattr(self, "frame_shift"):
                            num_frames = int(
                                self.utterances[utterance].duration * self.frame_shift
                            )
                        else:
                            num_frames = self.utterances[utterance].duration
                        self.utterances[utterance].alignment_log_likelihood = (
                            log_likelihood / num_frames
                        )
                    pbar.update(1)
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                self.logger.debug("Not using multiprocessing...")
                for args in self.align_arguments():
                    function = AlignFunction(args)
                    for utterance, log_likelihood in function.run():
                        if hasattr(self, "utterances"):
                            if hasattr(self, "frame_shift"):
                                num_frames = int(
                                    self.utterances[utterance].duration * self.frame_shift
                                )
                            else:
                                num_frames = self.utterances[utterance].duration
                            self.utterances[utterance].alignment_log_likelihood = (
                                log_likelihood / num_frames
                            )
                        pbar.update(1)

        self.compile_information()
        self.logger.debug(f"Alignment round took {time.time() - begin}")

    def compile_information(self):
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

        if self.use_mp:
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

        if hasattr(self, "utterances"):
            csv_path = os.path.join(self.working_directory, "alignment_log_likelihood.csv")
            with open(csv_path, "w", newline="", encoding="utf8") as f:
                writer = csv.writer(f)
                writer.writerow(["utterance", "loglikelihood"])
                for u in self.utterances:
                    writer.writerow([u.name, u.alignment_log_likelihood])

        if not avg_like_frames:
            self.logger.warning(
                "No files were aligned, this likely indicates serious problems with the aligner."
            )
        else:
            if too_short_count:
                self.logger.debug(
                    f"There were {too_short_count} utterances that were too short to be aligned."
                )
            if beam_too_narrow_count:
                self.logger.debug(
                    f"There were {beam_too_narrow_count} utterances that could not be aligned with "
                    f"the current beam settings."
                )
            average_log_like = avg_like_sum / avg_like_frames
            if average_logdet_sum:
                average_log_like += average_logdet_sum / average_logdet_frames
            self.logger.debug(f"Average per frame likelihood for alignment: {average_log_like}")
        self.logger.debug(f"Compiling information took {time.time() - compile_info_begin}")

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
    def alignment_model_path(self) -> str:
        """Acoustic model file path for speaker-independent alignment"""
        path = os.path.join(self.working_directory, "final.alimdl")
        if os.path.exists(path):
            return path
        return self.model_path
