"""Class definitions for alignment mixins"""
from __future__ import annotations

import datetime
import logging
import os
import time
import typing
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, List

from montreal_forced_aligner import config
from montreal_forced_aligner.alignment.multiprocessing import (
    AlignArguments,
    AlignFunction,
    CompileTrainGraphsArguments,
    CompileTrainGraphsFunction,
    PhoneConfidenceArguments,
    PhoneConfidenceFunction,
)
from montreal_forced_aligner.db import CorpusWorkflow, Job, PhoneInterval, Utterance, bulk_update
from montreal_forced_aligner.dictionary.mixins import DictionaryMixin
from montreal_forced_aligner.exceptions import NoAlignmentsError
from montreal_forced_aligner.utils import run_kaldi_function

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
        self.final_alignment = False

    @property
    def tree_path(self) -> Path:
        """Path to tree file"""
        return self.working_directory.joinpath("tree")

    @property
    @abstractmethod
    def data_directory(self) -> str:
        """Corpus data directory"""
        ...

    def compile_train_graphs_arguments(self) -> typing.List[CompileTrainGraphsArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.CompileTrainGraphsFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.CompileTrainGraphsArguments`]
            Arguments for processing
        """

        args = []
        lexicon_compilers = {}
        if getattr(self, "use_g2p", False):
            lexicon_compilers = getattr(self, "lexicon_compilers", {})
        for j in self.jobs:
            args.append(
                CompileTrainGraphsArguments(
                    j.id,
                    getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                    self.working_log_directory.joinpath(f"compile_train_graphs.{j.id}.log"),
                    self.working_directory,
                    lexicon_compilers,
                    self.working_directory.joinpath("tree"),
                    self.alignment_model_path,
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
                log_path = self.working_log_directory.joinpath(f"align.{iteration}.{j.id}.log")
            else:
                log_path = self.working_log_directory.joinpath(f"align.{j.id}.log")
            if getattr(self, "uses_speaker_adaptation", False):
                log_path = log_path.with_suffix(".fmllr.log")
            args.append(
                AlignArguments(
                    j.id,
                    getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                    log_path,
                    self.working_directory,
                    self.alignment_model_path,
                    self.align_options,
                    self.phone_confidence,
                    getattr(self, "final_alignment", False),
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
            log_path = self.working_log_directory.joinpath(f"phone_confidence.{j.id}.log")
            args.append(
                PhoneConfidenceArguments(
                    j.id,
                    getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                    log_path,
                    self.working_directory,
                    self.model_path,
                    self.phone_pdf_counts_path,
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
        arguments = self.compile_train_graphs_arguments()
        for _ in run_kaldi_function(CompileTrainGraphsFunction, arguments):
            pass
        logger.debug(f"Compiling training graphs took {time.time() - begin:.3f} seconds")

    def get_phone_confidences(self):
        if not os.path.exists(self.phone_pdf_counts_path):
            logger.warning("Cannot calculate phone confidences with the current model.")
            return
        logger.info("Calculating phone confidences...")
        begin = time.time()

        with self.session() as session:
            arguments = self.phone_confidence_arguments()
            interval_update_mappings = []
            for result in run_kaldi_function(
                PhoneConfidenceFunction, arguments, total_count=self.num_current_utterances
            ):
                interval_update_mappings.extend(result)
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
        self.working_log_directory.mkdir(parents=True, exist_ok=True)
        log_like_sum = 0
        log_like_count = 0
        update_mappings = []
        num_errors = 0
        num_successful = 0
        for utterance, log_likelihood in run_kaldi_function(
            AlignFunction, self.align_arguments(), total_count=self.num_current_utterances
        ):
            if log_likelihood:
                num_successful += 1
                if log_likelihood:
                    log_like_sum += log_likelihood
                    log_like_count += 1
            else:
                num_errors += 1
            if not training:
                update_mappings.append(
                    {
                        "id": int(utterance.split("-")[-1]),
                        "alignment_log_likelihood": log_likelihood,
                    }
                )
        if not training:
            if len(update_mappings) == 0 or num_successful == 0:
                raise NoAlignmentsError(self.num_current_utterances, self.beam, self.retry_beam)
            with self.session() as session:
                bulk_update(session, Utterance, update_mappings)
                session.commit()
                session.query(Utterance).filter(
                    Utterance.alignment_log_likelihood != None  # noqa
                ).update(
                    {
                        Utterance.alignment_log_likelihood: Utterance.alignment_log_likelihood
                        / Utterance.num_frames
                    },
                    synchronize_session="fetch",
                )
                workflow = (
                    session.query(CorpusWorkflow)
                    .filter(CorpusWorkflow.current == True)  # noqa
                    .first()
                )
                workflow.time_stamp = datetime.datetime.now()
                workflow.score = log_like_sum / log_like_count
                session.commit()
        logger.debug(
            f"Aligned {num_successful}, errors on {num_errors}, total {num_successful + num_errors}"
        )
        logger.debug(f"Alignment round took {time.time() - begin:.3f} seconds")

    @property
    @abstractmethod
    def working_directory(self) -> Path:
        """Working directory"""
        ...

    @property
    @abstractmethod
    def working_log_directory(self) -> Path:
        """Working log directory"""
        ...

    @property
    def model_path(self) -> Path:
        """Acoustic model file path"""
        return self.working_directory.joinpath("final.mdl")

    @property
    def phone_pdf_counts_path(self) -> Path:
        """Acoustic model file path"""
        return self.working_directory.joinpath("phone_pdf.counts")

    @property
    def alignment_model_path(self) -> Path:
        """Acoustic model file path for speaker-independent alignment"""
        path = self.working_directory.joinpath("final.alimdl")
        if os.path.exists(path) and not getattr(self, "uses_speaker_adaptation", False):
            return path
        return self.model_path
