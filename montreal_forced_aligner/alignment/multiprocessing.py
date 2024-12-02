"""
Alignment multiprocessing functions
-----------------------------------
"""
from __future__ import annotations

import collections
import json
import logging
import math
import multiprocessing as mp
import os
import shutil
import statistics
import sys
import time
import traceback
import typing
from pathlib import Path
from queue import Empty
from typing import TYPE_CHECKING

import numpy as np
import sqlalchemy
from _kalpy import feat as kalpy_feat
from _kalpy import transform as kalpy_transform
from _kalpy.gmm import gmm_compute_likes
from _kalpy.hmm import TransitionModel
from _kalpy.matrix import FloatMatrix, FloatSubMatrix
from _kalpy.util import RandomAccessBaseDoubleMatrixReader, RandomAccessBaseFloatMatrixReader
from kalpy.data import Segment
from kalpy.decoder.data import FstArchive
from kalpy.decoder.training_graphs import TrainingGraphCompiler
from kalpy.feat.mfcc import MfccComputer
from kalpy.feat.pitch import PitchComputer
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.gmm.align import GmmAligner
from kalpy.gmm.data import AlignmentArchive, TranscriptionArchive
from kalpy.gmm.train import GmmStatsAccumulator
from kalpy.gmm.utils import read_gmm_model
from kalpy.utils import generate_read_specifier, read_kaldi_object
from sqlalchemy.orm import joinedload, selectinload, subqueryload

from montreal_forced_aligner.abc import KaldiFunction
from montreal_forced_aligner.data import (
    WORD_BEGIN_SYMBOL,
    WORD_END_SYMBOL,
    MfaArguments,
    PhoneType,
    PronunciationProbabilityCounter,
    WordType,
    WorkflowType,
)
from montreal_forced_aligner.db import (
    CorpusWorkflow,
    File,
    Job,
    Phone,
    PhoneInterval,
    SoundFile,
    Speaker,
    TextFile,
    Utterance,
    Word,
    WordInterval,
)
from montreal_forced_aligner.exceptions import AlignmentCollectionError, AlignmentExportError
from montreal_forced_aligner.helper import (
    align_words,
    fix_unk_words,
    mfa_open,
    split_phone_position,
)
from montreal_forced_aligner.textgrid import construct_textgrid_output
from montreal_forced_aligner.utils import thread_logger

if TYPE_CHECKING:
    from dataclasses import dataclass

    from montreal_forced_aligner.abc import MetaDict
else:
    from dataclassy import dataclass


__all__ = [
    "AlignmentExtractionFunction",
    "ExportTextGridProcessWorker",
    "AlignmentExtractionArguments",
    "ExportTextGridArguments",
    "AlignFunction",
    "AlignArguments",
    "AnalyzeAlignmentsFunction",
    "AnalyzeAlignmentsArguments",
    "AnalyzeTranscriptsFunction",
    "AccStatsFunction",
    "AccStatsArguments",
    "CompileTrainGraphsFunction",
    "CompileTrainGraphsArguments",
    "GeneratePronunciationsArguments",
    "GeneratePronunciationsFunction",
    "FineTuneArguments",
    "FineTuneFunction",
    "PhoneConfidenceArguments",
    "PhoneConfidenceFunction",
]

logger = logging.getLogger("mfa")


@dataclass
class GeneratePronunciationsArguments(MfaArguments):
    """
    Arguments for :func:`~montreal_forced_aligner.alignment.multiprocessing.GeneratePronunciationsFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    session: :class:`sqlalchemy.orm.scoped_session` or str
        SqlAlchemy scoped session or string for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
    aligner: :class:`kalpy.gmm.align.GmmAligner`
        GmmAligner to use
    lexicon_compilers: dict[int, :class:`kalpy.fstext.lexicon.LexiconCompiler`]
        Lexicon compilers for each pronunciation dictionary
    for_g2p: bool
        Flag for training a G2P model with acoustic information
    """

    aligner: GmmAligner
    lexicon_compilers: typing.Dict[int, LexiconCompiler]
    for_g2p: bool


@dataclass
class AlignmentExtractionArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.AlignmentExtractionFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    session: :class:`sqlalchemy.orm.scoped_session` or str
        SqlAlchemy scoped session or string for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
    working_directory: :class:`~pathlib.Path`
        Working directory
    lexicon_compilers: dict[int, :class:`kalpy.fstext.lexicon.LexiconCompiler`]
        Lexicon compilers for each pronunciation dictionary
    aligner: :class:`kalpy.gmm.align.GmmAligner`
        GmmAligner to use
    frame_shift: float
        Frame shift in seconds
    ali_paths: dict[int, Path]
        Per dictionary alignment paths
    text_int_paths: dict[int, Path]
        Per dictionary text SCP paths
    phone_symbol_path: :class:`~pathlib.Path`
        Path to phone symbols table
    score_options: dict[str, Any]
        Options for Kaldi functions
    """

    working_directory: Path
    lexicon_compilers: typing.Dict[int, LexiconCompiler]
    transition_model: TransitionModel
    frame_shift: float
    score_options: MetaDict
    confidence: bool
    transcription: bool
    use_g2p: bool


@dataclass
class ExportTextGridArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.ExportTextGridProcessWorker`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    session: :class:`sqlalchemy.orm.scoped_session` or str
        SqlAlchemy scoped session or string for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
    export_frame_shift: float
        Frame shift in seconds
    cleanup_textgrids: bool
        Flag to cleanup silences and recombine words
    clitic_marker: str
        Marker indicating clitics
    output_directory: :class:`~pathlib.Path`
        Directory for exporting
    output_format: str
        Format to export
    include_original_text: bool
        Flag for including original unnormalized text as a tier
    """

    export_frame_shift: float
    cleanup_textgrids: bool
    clitic_marker: str
    output_directory: Path
    output_format: str
    include_original_text: bool


@dataclass
class CompileTrainGraphsArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.CompileTrainGraphsFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    session: :class:`sqlalchemy.orm.scoped_session` or str
        SqlAlchemy scoped session or string for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
    working_directory: :class:`~pathlib.Path`
        Working directory
    tree_path: :class:`~pathlib.Path`
        Path to tree file
    model_path: :class:`~pathlib.Path`
        Path to model file
    use_g2p: bool
        Flag for whether acoustic model uses g2p
    """

    working_directory: Path
    lexicon_compilers: typing.Dict[int, LexiconCompiler]
    tree_path: Path
    model_path: Path
    use_g2p: bool


@dataclass
class AlignArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.AlignFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    session: :class:`sqlalchemy.orm.scoped_session` or str
        SqlAlchemy scoped session or string for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
    working_directory: :class:`~pathlib.Path`
        Working directory
    model_path: :class:`~pathlib.Path`
        Path to model file
    align_options: dict[str, Any]
        Alignment options
    confidence: bool
        Flag for outputting confidence
    """

    working_directory: Path
    model_path: Path
    align_options: MetaDict
    confidence: bool
    final: bool


@dataclass
class AnalyzeAlignmentsArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.AnalyzeAlignmentsFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    session: :class:`sqlalchemy.orm.scoped_session` or str
        SqlAlchemy scoped session or string for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
    model_path: :class:`~pathlib.Path`
        Path to model file
    align_options: dict[str, Any]
        Alignment options
    """

    model_path: Path
    align_options: MetaDict


@dataclass
class FineTuneArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.AlignFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    session: :class:`sqlalchemy.orm.scoped_session` or str
        SqlAlchemy scoped session or string for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
    tree_path: :class:`~pathlib.Path`
        Path to tree file
    model_path: :class:`~pathlib.Path`
        Path to model file
    frame_shift: int
        Frame shift in ms
    mfcc_options: dict[str, Any]
        MFCC computation options
    pitch_options: dict[str, Any]
        Pitch computation options
    align_options: dict[str, Any]
        Alignment options
    position_dependent_phones: bool
        Flag for whether to use position dependent phones
    grouped_phones: dict[str, list[str]]
        Grouped lists of phones
    """

    mfcc_computer: MfccComputer
    pitch_computer: typing.Optional[PitchComputer]
    lexicon_compiler: LexiconCompiler
    model_path: Path
    tree_path: Path
    align_options: MetaDict
    phone_to_group_mapping: typing.Dict[str, str]
    original_frame_shift: float


@dataclass
class PhoneConfidenceArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.AlignFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    session: :class:`sqlalchemy.orm.scoped_session` or str
        SqlAlchemy scoped session or string for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
    working_directory: :class:`~pathlib.Path`
        Working directory
    model_path: :class:`~pathlib.Path`
        Path to model file
    phone_pdf_counts_path: :class:`~pathlib.Path`
        Path to output PDF counts
    """

    working_directory: Path
    model_path: Path
    phone_pdf_counts_path: Path


@dataclass
class AccStatsArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.AccStatsFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    session: :class:`sqlalchemy.orm.scoped_session` or str
        SqlAlchemy scoped session or string for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
    working_directory: :class:`~pathlib.Path`
        Path to working directory
    model_path: :class:`~pathlib.Path`
        Path to model file
    """

    working_directory: Path
    model_path: Path


class CompileTrainGraphsFunction(KaldiFunction):
    """
    Multiprocessing function to compile training graphs

    See Also
    --------
    :meth:`.AlignMixin.compile_train_graphs`
        Main function that calls this function in parallel
    :meth:`.AlignMixin.compile_train_graphs_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`compile-train-graphs`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.alignment.multiprocessing.CompileTrainGraphsArguments`
        Arguments for the function
    """

    def __init__(self, args: CompileTrainGraphsArguments):
        super().__init__(args)
        self.tree_path = args.tree_path
        self.lexicon_compilers = args.lexicon_compilers
        self.model_path = args.model_path
        self.use_g2p = args.use_g2p

    def _run(self):
        """Run the function"""

        with self.session() as session, thread_logger(
            "kalpy.graphs", self.log_path, job_name=self.job_name
        ) as graph_logger:
            graph_logger.debug(f"Tree path: {self.tree_path}")
            graph_logger.debug(f"Model path: {self.model_path}")
            job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )
            workflow: CorpusWorkflow = (
                session.query(CorpusWorkflow)
                .filter(CorpusWorkflow.current == True)  # noqa
                .first()
            )
            interjection_costs = {}
            if workflow.workflow_type is WorkflowType.transcript_verification:
                interjection_words = (
                    session.query(Word).filter(Word.word_type == WordType.interjection).all()
                )
                if interjection_words:
                    max_count = max(math.log(x.count) for x in interjection_words)
                    for w in interjection_words:
                        count = math.log(w.count)
                        if count == 0:
                            count = 0.01
                        cost = max_count / count
                        interjection_costs[w.word] = cost
            if self.use_g2p:
                text_column = Utterance.normalized_character_text
            else:
                text_column = Utterance.normalized_text
            for d in job.training_dictionaries:
                begin = time.time()
                if self.lexicon_compilers and d.id in self.lexicon_compilers:
                    lexicon = self.lexicon_compilers[d.id]
                else:
                    lexicon = d.lexicon_compiler
                if workflow.workflow_type is WorkflowType.transcript_verification:
                    if interjection_words and d.oov_word not in interjection_costs:
                        interjection_costs[d.oov_word] = min(interjection_costs.values())
                        # interjection_costs[d.cutoff_word] = min(interjection_costs.values())
                compiler = TrainingGraphCompiler(
                    self.model_path,
                    self.tree_path,
                    lexicon,
                    use_g2p=self.use_g2p,
                    batch_size=500
                    if workflow.workflow_type is not WorkflowType.transcript_verification
                    else 250,
                )
                graph_logger.debug(f"Set up took {time.time() - begin} seconds")
                query = (
                    session.query(Utterance.kaldi_id, text_column)
                    .join(Utterance.speaker)
                    .filter(Utterance.job_id == self.job_name, Speaker.dictionary_id == d.id)
                    .filter(Utterance.ignored == False)  # noqa
                    .order_by(Utterance.kaldi_id)
                )
                if job.corpus.current_subset > 0:
                    query = query.filter(Utterance.in_subset == True)  # noqa
                graph_logger.info(f"Compiling graphs for {d.name}")
                fst_ark_path = job.construct_path(workflow.working_directory, "fsts", "ark", d.id)
                compiler.export_graphs(
                    fst_ark_path,
                    query,
                    # callback=self.callback,
                    interjection_words=interjection_costs,
                    # cutoff_pattern = d.cutoff_word
                )
                graph_logger.debug(f"Total compilation time: {time.time() - begin} seconds")
                del compiler
                del lexicon


class AccStatsFunction(KaldiFunction):
    """
    Multiprocessing function for accumulating stats in GMM training.

    See Also
    --------
    :meth:`.AcousticModelTrainingMixin.acc_stats`
        Main function that calls this function in parallel
    :meth:`.AcousticModelTrainingMixin.acc_stats_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`gmm-acc-stats-ali`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.alignment.multiprocessing.AccStatsArguments`
        Arguments for the function
    """

    def __init__(self, args: AccStatsArguments):
        super().__init__(args)
        self.working_directory = args.working_directory
        self.model_path = args.model_path

    def _run(self) -> None:
        """Run the function"""
        with self.session() as session, thread_logger(
            "kalpy.train", self.log_path, job_name=self.job_name
        ) as train_logger:
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )
            for d in job.training_dictionaries:
                train_logger.debug(f"Accumulating stats for dictionary {d.name} ({d.id})")
                train_logger.debug(f"Accumulating stats for model: {self.model_path}")
                dict_id = d.id
                accumulator = GmmStatsAccumulator(self.model_path)

                feature_archive = job.construct_feature_archive(self.working_directory, dict_id)
                ali_path = job.construct_path(self.working_directory, "ali", "ark", dict_id)
                if not ali_path.exists():
                    continue
                alignment_archive = AlignmentArchive(ali_path)
                train_logger.debug("Feature Archive information:")
                train_logger.debug(f"CMVN: {feature_archive.cmvn_read_specifier}")
                train_logger.debug(f"Deltas: {feature_archive.use_deltas}")
                train_logger.debug(f"Splices: {feature_archive.use_splices}")
                train_logger.debug(f"LDA: {feature_archive.lda_mat_file_name}")
                train_logger.debug(f"fMLLR: {feature_archive.transform_read_specifier}")
                train_logger.debug(f"Alignment path: {ali_path}")

                accumulator.accumulate_stats(
                    feature_archive, alignment_archive, callback=self.callback
                )
                self.callback((accumulator.transition_accs, accumulator.gmm_accs))


class AlignFunction(KaldiFunction):
    """
    Multiprocessing function for alignment.

    See Also
    --------
    :meth:`.AlignMixin.align_utterances`
        Main function that calls this function in parallel
    :meth:`.AlignMixin.align_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`align-gmm-compiled`
        Relevant Kaldi binary
    :kaldi_src:`gmm-boost-silence`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.alignment.multiprocessing.AlignArguments`
        Arguments for the function
    """

    def __init__(self, args: AlignArguments):
        super().__init__(args)
        self.working_directory = args.working_directory
        self.model_path = args.model_path
        self.align_options = args.align_options
        self.confidence = args.confidence
        self.final = args.final

    def _run(self) -> None:
        """Run the function"""
        with self.session() as session, thread_logger(
            "kalpy.align", self.log_path, job_name=self.job_name
        ) as align_logger:
            align_logger.debug(f"Align options: {self.align_options}")
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )
            align_options = self.align_options
            boost_silence = align_options.pop("boost_silence", 1.0)
            silence_phones = [
                x
                for x, in session.query(Phone.mapping_id).filter(
                    Phone.phone_type == PhoneType.silence, Phone.phone != "<eps>"
                )
            ]
            aligner = GmmAligner(
                self.model_path,
                **align_options,
            )
            aligner.boost_silence(boost_silence, silence_phones)
            for d in job.training_dictionaries:
                align_logger.debug(f"Aligning for dictionary {d.name} ({d.id})")
                align_logger.debug(f"Aligning with model: {aligner.acoustic_model_path}")
                dict_id = d.id
                fst_path = job.construct_path(self.working_directory, "fsts", "ark", dict_id)
                align_logger.debug(f"Training graph archive: {fst_path}")
                feature_archive = job.construct_feature_archive(self.working_directory, dict_id)

                align_logger.debug("Feature Archive information:")
                align_logger.debug(f"CMVN: {feature_archive.cmvn_read_specifier}")
                align_logger.debug(f"Deltas: {feature_archive.use_deltas}")
                align_logger.debug(f"Splices: {feature_archive.use_splices}")
                align_logger.debug(f"LDA: {feature_archive.lda_mat_file_name}")
                align_logger.debug(f"fMLLR: {feature_archive.transform_read_specifier}")

                training_graph_archive = FstArchive(fst_path)
                ali_path = job.construct_path(self.working_directory, "ali", "ark", dict_id)

                words_path = job.construct_path(self.working_directory, "words", "ark", dict_id)
                likes_path = job.construct_path(
                    self.working_directory, "likelihoods", "ark", dict_id
                )
                ali_path.unlink(missing_ok=True)
                words_path.unlink(missing_ok=True)
                likes_path.unlink(missing_ok=True)
                if aligner.acoustic_model_path.endswith(".alimdl"):
                    ali_path = job.construct_path(
                        self.working_directory, "ali_first_pass", "ark", dict_id
                    )
                    words_path = job.construct_path(
                        self.working_directory, "words_first_pass", "ark", dict_id
                    )
                    likes_path = job.construct_path(
                        self.working_directory, "likelihoods_first_pass", "ark", dict_id
                    )
                aligner.export_alignments(
                    ali_path,
                    training_graph_archive,
                    feature_archive,
                    word_file_name=words_path,
                    likelihood_file_name=likes_path,
                    callback=self.callback,
                )
                if aligner.acoustic_model_path.endswith(".alimdl"):
                    try:
                        job.construct_path(
                            self.working_directory, "ali", "ark", dict_id
                        ).symlink_to(ali_path)
                        job.construct_path(
                            self.working_directory, "words", "ark", dict_id
                        ).symlink_to(words_path)
                        job.construct_path(
                            self.working_directory, "likelihoods", "ark", dict_id
                        ).symlink_to(likes_path)
                    except OSError:
                        shutil.copyfile(
                            ali_path,
                            job.construct_path(self.working_directory, "ali", "ark", dict_id),
                        )
                        shutil.copyfile(
                            words_path,
                            job.construct_path(self.working_directory, "words", "ark", dict_id),
                        )
                        shutil.copyfile(
                            likes_path,
                            job.construct_path(
                                self.working_directory, "likelihoods", "ark", dict_id
                            ),
                        )


class AnalyzeAlignmentsFunction(KaldiFunction):
    """
    Multiprocessing function for analyzing alignments.

    See Also
    --------
    :meth:`.CorpusAligner.analyze_alignments`
        Main function that calls this function in parallel
    :meth:`.CorpusAligner.calculate_speech_post_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`lattice-to-post`
        Relevant Kaldi binary
    :kaldi_src:`weight-silence-post`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.alignment.multiprocessing.CalculateSpeechPostArguments`
        Arguments for the function
    """

    def __init__(self, args: AnalyzeAlignmentsArguments):
        super().__init__(args)
        self.model_path = args.model_path
        self.align_options = args.align_options

    def _run(self):
        """Run the function"""

        with self.session() as session:
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )
            workflow = (
                session.query(CorpusWorkflow)
                .filter(CorpusWorkflow.current == True)  # noqa
                .first()
            )
            phones = {
                k: (m, sd)
                for k, m, sd in session.query(
                    Phone.id, Phone.mean_duration, Phone.sd_duration
                ).filter(
                    Phone.phone_type == PhoneType.non_silence,
                    Phone.sd_duration != None,  # noqa
                    Phone.sd_duration != 0,
                )
            }
            query = session.query(Utterance).filter(
                Utterance.job_id == job.id, Utterance.alignment_log_likelihood != None  # noqa
            )
            for utterance in query:
                phone_intervals = (
                    session.query(PhoneInterval)
                    .join(PhoneInterval.phone)
                    .filter(
                        PhoneInterval.utterance_id == utterance.id,
                        PhoneInterval.workflow_id == workflow.id,
                        Phone.id.in_(list(phones.keys())),
                    )
                    .all()
                )
                if not phone_intervals:
                    continue
                interval_count = len(phone_intervals)
                log_like_sum = 0
                duration_zscore_max = 0
                for pi in phone_intervals:
                    log_like_sum += pi.phone_goodness
                    m, sd = phones[pi.phone_id]
                    duration_zscore = abs((pi.duration - m) / sd)
                    if duration_zscore > duration_zscore_max:
                        duration_zscore_max = duration_zscore
                utterance_speech_log_likelihood = log_like_sum / interval_count
                utterance_duration_deviation = duration_zscore_max
                self.callback(
                    (utterance.id, utterance_speech_log_likelihood, utterance_duration_deviation)
                )


class AnalyzeTranscriptsFunction(KaldiFunction):
    """
    Multiprocessing function for analyzing alignments.

    See Also
    --------
    :meth:`.CorpusAligner.analyze_alignments`
        Main function that calls this function in parallel
    :meth:`.CorpusAligner.calculate_speech_post_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`lattice-to-post`
        Relevant Kaldi binary
    :kaldi_src:`weight-silence-post`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.alignment.multiprocessing.CalculateSpeechPostArguments`
        Arguments for the function
    """

    def __init__(self, args: AnalyzeAlignmentsArguments):
        super().__init__(args)
        self.model_path = args.model_path
        self.align_options = args.align_options

    def _run(self):
        """Run the function"""

        with self.session() as session:
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )
            workflow = (
                session.query(CorpusWorkflow)
                .filter(CorpusWorkflow.current == True)  # noqa
                .first()
            )
            query = session.query(Utterance).filter(
                Utterance.job_id == job.id, Utterance.alignment_log_likelihood != None  # noqa
            )
            for utterance in query:
                word_intervals = [
                    x.as_ctm()
                    for x in (
                        session.query(WordInterval)
                        .join(WordInterval.word)
                        .filter(
                            WordInterval.utterance_id == utterance.id,
                            WordInterval.workflow_id == workflow.id,
                            Word.word_type != WordType.silence,
                            WordInterval.end - WordInterval.begin > 0.03,
                        )
                        .options(
                            joinedload(WordInterval.word, innerjoin=True),
                        )
                        .order_by(WordInterval.begin)
                    )
                ]
                if not word_intervals:
                    continue
                extra_duration, wer, aligned_duration = align_words(
                    utterance.normalized_text.split(), word_intervals, "<eps>", debug=True
                )
                transcript = " ".join(x.label for x in word_intervals)
                self.callback((utterance.id, wer, extra_duration, transcript))


class FineTuneFunction(KaldiFunction):
    """
    Multiprocessing function for fine tuning alignment.

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.alignment.multiprocessing.FineTuneArguments`
        Arguments for the function
    """

    def __init__(self, args: FineTuneArguments):
        super().__init__(args)
        self.mfcc_computer = args.mfcc_computer
        self.pitch_computer = args.pitch_computer
        self.lexicon_compiler = args.lexicon_compiler
        self.model_path = args.model_path
        self.tree_path = args.tree_path
        self.align_options = args.align_options
        self.phone_to_group_mapping = args.phone_to_group_mapping
        self.frame_shift_seconds = args.original_frame_shift

        self.new_frame_shift_seconds = 0.001
        self.feature_padding_factor = 3
        self.padding = round(self.frame_shift_seconds, 3)
        self.splice_frames = 3

    def _run(self):
        """Run the function"""
        with self.session() as session, thread_logger(
            "kalpy.align", self.log_path, job_name=self.job_name
        ):
            job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )
            workflow: CorpusWorkflow = (
                session.query(CorpusWorkflow)
                .filter(CorpusWorkflow.current == True)  # noqa
                .first()
            )

            phone_mapping = {}
            phone_query = session.query(Phone.id, Phone.kaldi_label)
            for p_id, phone in phone_query:
                phone_mapping[phone] = p_id
            disambiguation_symbols = [
                x
                for x, in session.query(Phone.mapping_id).filter(
                    Phone.phone_type == PhoneType.disambiguation
                )
            ]
            silence_phones = [
                x
                for x, in session.query(Phone.mapping_id).filter(
                    Phone.phone_type.in_([PhoneType.silence])
                )
            ]

            cmvn_paths = job.per_dictionary_cmvn_scp_paths
            trans_paths = job.per_dictionary_trans_scp_paths
            compiler = TrainingGraphCompiler(
                self.model_path,
                self.tree_path,
                self.lexicon_compiler,
            )
            for d_id in job.dictionary_ids:
                utterance_query = (
                    session.query(Utterance, SoundFile.sound_file_path)
                    .join(Utterance.file)
                    .join(Utterance.speaker)
                    .join(File.sound_file)
                    .filter(Utterance.job_id == self.job_name, Speaker.dictionary_id == d_id)
                    .order_by(Utterance.kaldi_id)
                )
                boost_silence = self.align_options.pop("boost_silence", 1.0)
                aligner = GmmAligner(
                    self.model_path,
                    disambiguation_symbols=disambiguation_symbols,
                    **self.align_options,
                )
                if boost_silence != 1.0:
                    aligner.boost_silence(boost_silence, silence_phones)
                delta_options = kalpy_feat.DeltaFeaturesOptions()
                use_splices = False
                use_deltas = True
                lda_mat = None
                if workflow.lda_mat_path.exists():
                    use_splices = True
                    use_deltas = False
                    lda_mat = read_kaldi_object(FloatMatrix, workflow.lda_mat_path)

                cmvn_reader = None
                cmvn_path = cmvn_paths[d_id]
                if cmvn_path.exists():
                    cmvn_read_specifier = generate_read_specifier(cmvn_path)
                    cmvn_reader = RandomAccessBaseDoubleMatrixReader(cmvn_read_specifier)

                fmllr_path = trans_paths[d_id]
                transform_reader = None
                if fmllr_path.exists():
                    transform_read_specifier = generate_read_specifier(fmllr_path)
                    transform_reader = RandomAccessBaseFloatMatrixReader(transform_read_specifier)
                current_speaker = None
                current_transform = None
                current_cmvn = None
                for utterance, sf_path in utterance_query:
                    interval_query = (
                        session.query(PhoneInterval, Phone.kaldi_label)
                        .join(PhoneInterval.phone)
                        .filter(
                            PhoneInterval.utterance_id == utterance.id,
                            PhoneInterval.workflow_id == workflow.id,
                        )
                        .order_by(PhoneInterval.begin)
                    )
                    prev_label = None
                    if utterance.speaker_id != current_speaker:
                        current_speaker = utterance.speaker_id
                        if cmvn_reader is not None and cmvn_reader.HasKey(str(current_speaker)):
                            current_cmvn = cmvn_reader.Value(str(current_speaker))
                        if transform_reader is not None and transform_reader.HasKey(
                            str(current_speaker)
                        ):
                            current_transform = transform_reader.Value(str(current_speaker))
                    interval_mapping = []
                    for interval, phone in interval_query:
                        if prev_label is None:
                            prev_label = phone
                            interval_mapping.append(
                                {"id": interval.id, "begin": interval.begin, "end": interval.end}
                            )
                            continue
                        end_padding = round(self.frame_shift_seconds * 1.5, 3)
                        prev_padding = round(self.frame_shift_seconds * 1.5, 3)
                        segment_begin = max(round(interval.begin - prev_padding, 4), 0)
                        feature_segment_begin = max(
                            round(
                                interval.begin - (prev_padding * self.feature_padding_factor), 4
                            ),
                            0,
                        )
                        segment_end = round(min(interval.begin + end_padding, interval.end), 3)
                        feature_segment_end = min(
                            round(interval.begin + (end_padding * self.feature_padding_factor), 4),
                            utterance.end,
                        )
                        begin_offset = round(segment_begin - feature_segment_begin, 4)
                        end_offset = round(segment_end - feature_segment_begin, 4)
                        segment = Segment(
                            sf_path, feature_segment_begin, feature_segment_end, utterance.channel
                        )
                        text = f"{self.phone_to_group_mapping[prev_label]} {self.phone_to_group_mapping[phone]}"

                        train_graph = compiler.compile_fst(text)

                        feats = self.mfcc_computer.compute_mfccs_for_export(
                            segment, compress=False
                        )
                        if current_cmvn is not None:
                            kalpy_transform.ApplyCmvn(current_cmvn, False, feats)

                        if self.pitch_computer is not None:
                            pitch = self.pitch_computer.compute_pitch_for_export(
                                segment, compress=False
                            )
                            feats = kalpy_feat.paste_feats([feats, pitch], 0)
                        if use_deltas:
                            feats = kalpy_feat.compute_deltas(delta_options, feats)
                        elif use_splices:
                            feats = kalpy_feat.splice_frames(
                                feats, self.splice_frames, self.splice_frames
                            )
                            if lda_mat is not None:
                                feats = kalpy_transform.apply_transform(feats, lda_mat)
                        if current_transform is not None:
                            feats = kalpy_transform.apply_transform(feats, current_transform)
                        start_samp = int(round(begin_offset * 1000))
                        end_samp = int(round(end_offset * 1000))
                        sub_matrix = FloatSubMatrix(
                            feats, start_samp, end_samp - start_samp, 0, feats.NumCols()
                        )
                        feats = FloatMatrix(sub_matrix)
                        alignment = aligner.align_utterance(train_graph, feats)
                        if alignment is None:
                            aligner.acoustic_scale = 0.1
                            alignment = aligner.align_utterance(train_graph, feats)
                            aligner.acoustic_scale = 1.0
                        ctm_intervals = alignment.generate_ctm(
                            aligner.transition_model,
                            self.lexicon_compiler.phone_table,
                            frame_shift=0.001,
                        )
                        interval_mapping.append(
                            {
                                "id": interval.id,
                                "begin": round(
                                    ctm_intervals[1].begin + feature_segment_begin + begin_offset,
                                    4,
                                ),
                                "end": interval.end,
                                "label": phone_mapping[ctm_intervals[1].label],
                            }
                        )
                        prev_label = phone
                    deletions = []
                    while True:
                        for i in range(len(interval_mapping) - 1):
                            if interval_mapping[i]["end"] != interval_mapping[i + 1]["begin"]:
                                interval_mapping[i]["end"] = interval_mapping[i + 1]["begin"]
                        new_deletions = [
                            x["id"] for x in interval_mapping if x["begin"] >= x["end"]
                        ]
                        interval_mapping = [
                            x for x in interval_mapping if x["id"] not in new_deletions
                        ]
                        deletions.extend(new_deletions)
                        if not new_deletions and all(
                            interval_mapping[i]["end"] == interval_mapping[i + 1]["begin"]
                            for i in range(len(interval_mapping) - 1)
                        ):
                            break
                    self.callback((interval_mapping, deletions))


class PhoneConfidenceFunction(KaldiFunction):
    """
    Multiprocessing function to calculate phone confidence metrics

    See Also
    --------
    :kaldi_src:`gmm-compute-likes`
        Relevant Kaldi binary
    :kaldi_src:`transform-feats`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.alignment.multiprocessing.PhoneConfidenceArguments`
        Arguments for the function
    """

    def __init__(self, args: PhoneConfidenceArguments):
        super().__init__(args)
        self.working_directory = args.working_directory
        self.model_path = args.model_path
        self.phone_pdf_counts_path = args.phone_pdf_counts_path

    def _run(self):
        """Run the function"""

        with mfa_open(self.phone_pdf_counts_path, "r") as f:
            data = json.load(f)
        phone_pdf_mapping = collections.defaultdict(collections.Counter)
        for phone, pdf_counts in data.items():
            phone = split_phone_position(phone)[0]
            for pdf, count in pdf_counts.items():
                phone_pdf_mapping[phone][int(pdf)] += count
        phones = {p: i for i, p in enumerate(sorted(phone_pdf_mapping.keys()))}
        reversed_phones = {k: v for v, k in phones.items()}

        for phone, pdf_counts in phone_pdf_mapping.items():
            phone_total = sum(pdf_counts.values())
            for pdf, count in pdf_counts.items():
                phone_pdf_mapping[phone][int(pdf)] = count / phone_total
        _, acoustic_model = read_gmm_model(self.model_path)
        with self.session() as session:
            job: typing.Optional[Job] = session.get(
                Job, self.job_name, options=[joinedload(Job.dictionaries), joinedload(Job.corpus)]
            )
            utterances = (
                session.query(Utterance)
                .filter(Utterance.job_id == self.job_name)
                .options(
                    selectinload(Utterance.phone_intervals).joinedload(
                        PhoneInterval.phone, innerjoin=True
                    )
                )
            )
            utterances = {u.id: (u.begin, u.phone_intervals) for u in utterances}

            for dict_id in job.dictionary_ids:
                feature_archive = job.construct_feature_archive(self.working_directory, dict_id)
                interval_mappings = []

                for utterance_id, feats in feature_archive:
                    utterance_id = int(utterance_id.split("-")[-1])
                    likelihoods = gmm_compute_likes(acoustic_model, feats).numpy()
                    phone_likes = np.zeros((likelihoods.shape[0], len(phones)))
                    for i, p in reversed_phones.items():
                        like = likelihoods[:, [x for x in phone_pdf_mapping[p].keys()]]
                        weight = np.array([x for x in phone_pdf_mapping[p].values()])
                        phone_likes[:, i] = np.dot(like, weight)
                    top_phone_inds = np.argmax(phone_likes, axis=1)
                    utt_begin, intervals = utterances[utterance_id]
                    for pi in intervals:
                        if pi.phone.phone == "sil":
                            continue
                        frame_begin = int(((pi.begin - utt_begin) * 1000) / 10)
                        frame_end = int(((pi.end - utt_begin) * 1000) / 10)
                        if frame_begin == frame_end:
                            frame_end += 1
                        frame_end = min(frame_end, top_phone_inds.shape[0])
                        alternate_labels = collections.Counter()
                        scores = []

                        for i in range(frame_begin, frame_end):
                            top_phone_ind = top_phone_inds[i]
                            alternate_label = reversed_phones[top_phone_ind]
                            alternate_label = split_phone_position(alternate_label)[0]
                            alternate_labels[alternate_label] += 1
                            if alternate_label == pi.phone.phone:
                                scores.append(0)
                            else:
                                actual_score = phone_likes[i, phones[pi.phone.phone]]
                                scores.append(phone_likes[i, top_phone_ind] - actual_score)
                        average_score = statistics.mean(scores)
                        interval_mappings.append({"id": pi.id, "phone_goodness": float(average_score)})
                    self.callback(interval_mappings)
                    interval_mappings = []


class GeneratePronunciationsFunction(KaldiFunction):
    """
    Multiprocessing function for generating pronunciations

    See Also
    --------
    :meth:`.DictionaryTrainer.export_lexicons`
        Main function that calls this function in parallel
    :meth:`.CorpusAligner.generate_pronunciations_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`linear-to-nbest`
        Kaldi binary this uses

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.alignment.multiprocessing.AlignArguments`
        Arguments for the function
    """

    def __init__(self, args: GeneratePronunciationsArguments):
        super().__init__(args)
        self.aligner = args.aligner
        self.lexicon_compilers = args.lexicon_compilers
        self.for_g2p = args.for_g2p
        self.silence_words = set()

    def _process_pronunciations(
        self, word_pronunciations: typing.List[typing.Tuple[str, str]]
    ) -> PronunciationProbabilityCounter:
        """
        Process an utterance's pronunciations and extract relevant count information

        Parameters
        ----------
        word_pronunciations: list[tuple[str, tuple[str, ...]]]
            List of tuples containing the word integer ID and a list of the integer IDs of the phones
        """
        counter = PronunciationProbabilityCounter()
        word_pronunciations = [("<s>", "")] + word_pronunciations + [("</s>", "")]
        for i, w_p in enumerate(word_pronunciations):
            if i != 0:
                word = word_pronunciations[i - 1][0]
                if word in self.silence_words:
                    counter.silence_before_counts[w_p] += 1
                else:
                    counter.non_silence_before_counts[w_p] += 1
            silence_check = w_p[0] in self.silence_words
            if not silence_check:
                counter.word_pronunciation_counts[w_p[0]][w_p[1]] += 1
                if i != len(word_pronunciations) - 1:
                    word = word_pronunciations[i + 1][0]
                    if word in self.silence_words:
                        counter.silence_following_counts[w_p] += 1
                        if i != len(word_pronunciations) - 2:
                            next_w_p = word_pronunciations[i + 2]
                            counter.ngram_counts[w_p, next_w_p]["silence"] += 1
                    else:
                        next_w_p = word_pronunciations[i + 1]
                        counter.non_silence_following_counts[w_p] += 1
                        counter.ngram_counts[w_p, next_w_p]["non_silence"] += 1
        return counter

    def _run(self) -> None:
        """Run the function"""
        with self.session() as session:
            job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )
            workflow: CorpusWorkflow = (
                session.query(CorpusWorkflow)
                .filter(CorpusWorkflow.current == True)  # noqa
                .first()
            )

            silence_words = session.query(Word.word).filter(Word.word_type == WordType.silence)
            self.silence_words.update(x for x, in silence_words)

            for d in job.training_dictionaries:
                ali_path = job.construct_path(workflow.working_directory, "ali", "ark", d.id)
                if not os.path.exists(ali_path):
                    continue
                if self.lexicon_compilers and d.id in self.lexicon_compilers:
                    lexicon_compiler = self.lexicon_compilers[d.id]
                else:
                    lexicon_compiler = d.lexicon_compiler

                words_path = job.construct_path(workflow.working_directory, "words", "ark", d.id)
                alignment_archive = AlignmentArchive(ali_path, words_file_name=words_path)
                for alignment in alignment_archive:
                    intervals = alignment.generate_ctm(
                        self.aligner.transition_model, lexicon_compiler.phone_table
                    )
                    utterance = int(alignment.utterance_id.split("-")[-1])
                    ctm = lexicon_compiler.phones_to_pronunciations(alignment.words, intervals)
                    word_pronunciations = []
                    for wi in ctm.word_intervals:
                        label = wi.label
                        pronunciation = wi.pronunciation
                        if label.startswith(d.cutoff_word[:-1]):
                            label = d.cutoff_word
                            if pronunciation != d.oov_phone:
                                pronunciation = "cutoff_model"
                        word_pronunciations.append((label, pronunciation))
                    if self.for_g2p:
                        phones = []
                        for i, x in enumerate(word_pronunciations):
                            if i > 0 and (
                                x[0].startswith(d.clitic_marker)
                                or word_pronunciations[i - 1][0].endswith(d.clitic_marker)
                            ):
                                phones.pop(-1)
                            else:
                                phones.append(WORD_BEGIN_SYMBOL)
                            phones.extend(x[1].split())
                            phones.append(WORD_END_SYMBOL)
                        self.callback((d.id, utterance, " ".join(phones)))
                    else:
                        self.callback((d.id, self._process_pronunciations(word_pronunciations)))


class AlignmentExtractionFunction(KaldiFunction):

    """
    Multiprocessing function to collect phone alignments from the aligned lattice

    See Also
    --------
    :meth:`.CorpusAligner.collect_alignments`
        Main function that calls this function in parallel
    :meth:`.CorpusAligner.alignment_extraction_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`linear-to-nbest`
        Relevant Kaldi binary
    :kaldi_src:`lattice-determinize-pruned`
        Relevant Kaldi binary
    :kaldi_src:`lattice-align-words`
        Relevant Kaldi binary
    :kaldi_src:`lattice-to-phone-lattice`
        Relevant Kaldi binary
    :kaldi_src:`nbest-to-ctm`
        Relevant Kaldi binary
    :kaldi_steps:`get_train_ctm`
        Reference Kaldi script

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.alignment.multiprocessing.AlignmentExtractionArguments`
        Arguments for the function
    """

    def __init__(self, args: AlignmentExtractionArguments):
        super().__init__(args)
        self.lexicon_compilers = args.lexicon_compilers
        self.working_directory = args.working_directory
        self.transition_model = args.transition_model
        self.frame_shift = args.frame_shift
        self.confidence = args.confidence
        self.transcription = args.transcription
        self.score_options = args.score_options
        self.use_g2p = args.use_g2p

    def _run(self) -> None:
        """Run the function"""
        with self.session() as session, thread_logger(
            "kalpy.align", self.log_path, job_name=self.job_name
        ) as extraction_logger:
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )
            workflow: CorpusWorkflow = (
                session.query(CorpusWorkflow)
                .filter(CorpusWorkflow.current == True)  # noqa
                .first()
            )

            for d in job.dictionaries:
                utterance_times = {}
                utterance_texts = {}
                if self.use_g2p:
                    utts = (
                        session.query(
                            Utterance.id,
                            Utterance.begin,
                            Utterance.end,
                            Utterance.normalized_character_text,
                        )
                        .join(Utterance.speaker)
                        .filter(Utterance.job_id == self.job_name)
                        .filter(Speaker.dictionary_id == d.id)
                    )
                    for u_id, begin, end, text in utts:
                        utterance_times[u_id] = (begin, end)
                        utterance_texts[u_id] = text

                else:
                    utts = (
                        session.query(
                            Utterance.id, Utterance.begin, Utterance.end, Utterance.normalized_text
                        )
                        .join(Utterance.speaker)
                        .filter(Utterance.job_id == self.job_name)
                        .filter(Speaker.dictionary_id == d.id)
                    )
                    for u_id, begin, end, text in utts:
                        utterance_times[u_id] = (begin, end)
                        utterance_texts[u_id] = text
                if self.lexicon_compilers and d.id in self.lexicon_compilers:
                    lexicon_compiler = self.lexicon_compilers[d.id]
                else:
                    lexicon_compiler = d.lexicon_compiler

                if self.transcription:
                    lat_path = job.construct_path(workflow.working_directory, "lat", "ark", d.id)
                    if not lat_path.exists():
                        continue

                    transcription_archive = TranscriptionArchive(
                        lat_path, acoustic_scale=self.score_options["acoustic_scale"]
                    )
                    for transcription in transcription_archive:
                        intervals = transcription.generate_ctm(
                            self.transition_model, lexicon_compiler.phone_table, self.frame_shift
                        )
                        utterance_id = int(transcription.utterance_id.split("-")[-1])
                        try:
                            ctm = lexicon_compiler.phones_to_pronunciations(
                                transcription.words,
                                intervals,
                                transcription=True,
                                text=utterance_texts.get(utterance_id, None),
                            )
                            ctm.update_utterance_boundaries(*utterance_times[utterance_id])
                        except Exception:
                            exc_type, exc_value, exc_traceback = sys.exc_info()
                            utterance, sound_file_path, text_file_path = (
                                session.query(
                                    Utterance, SoundFile.sound_file_path, TextFile.text_file_path
                                )
                                .join(Utterance.file)
                                .join(File.sound_file)
                                .join(File.text_file)
                                .filter(Utterance.id == utterance_id)
                                .first()
                            )
                            extraction_logger.debug(
                                f"Error processing {utterance} ({utterance_id}):"
                            )
                            extraction_logger.debug(
                                f"Utterance information: {sound_file_path}, {text_file_path}, {utterance.begin} - {utterance.end}"
                            )
                            traceback_lines = traceback.format_exception(
                                exc_type, exc_value, exc_traceback
                            )
                            extraction_logger.debug("\n".join(traceback_lines))
                            raise AlignmentCollectionError(
                                sound_file_path,
                                text_file_path,
                                utterance.begin,
                                utterance.end,
                                traceback_lines,
                                self.log_path,
                            )
                        self.callback((utterance_id, d.id, ctm))
                else:
                    ali_path = job.construct_path(workflow.working_directory, "ali", "ark", d.id)
                    if not ali_path.exists():
                        continue
                    words_path = job.construct_path(
                        workflow.working_directory, "words", "ark", d.id
                    )
                    likes_path = job.construct_path(
                        workflow.working_directory, "likelihoods", "ark", d.id
                    )
                    alignment_archive = AlignmentArchive(
                        ali_path, words_file_name=words_path, likelihood_file_name=likes_path
                    )
                    found_utterances = set()
                    for alignment in alignment_archive:
                        intervals = alignment.generate_ctm(
                            self.transition_model, lexicon_compiler.phone_table, self.frame_shift
                        )
                        utterance = int(alignment.utterance_id.split("-")[-1])
                        found_utterances.add(utterance)
                        try:
                            text = utterance_texts.get(utterance, None)
                            ctm = lexicon_compiler.phones_to_pronunciations(
                                alignment.words,
                                intervals,
                                transcription=False,
                                text=text,
                            )
                            ctm.update_utterance_boundaries(*utterance_times[utterance])
                            if text is not None:
                                ctm.word_intervals = fix_unk_words(
                                    text.split(), ctm.word_intervals, lexicon_compiler
                                )
                            extraction_logger.debug(f"Processed {utterance}")
                            self.callback((utterance, d.id, ctm))
                        except Exception:
                            exc_type, exc_value, exc_traceback = sys.exc_info()
                            utterance, sound_file_path, text_file_path = (
                                session.query(
                                    Utterance, SoundFile.sound_file_path, TextFile.text_file_path
                                )
                                .join(Utterance.file)
                                .join(File.sound_file)
                                .join(File.text_file)
                                .filter(Utterance.id == utterance)
                                .first()
                            )
                            extraction_logger.debug(f"Error processing {utterance}:")
                            extraction_logger.debug(
                                f"Utterance information: {sound_file_path}, {text_file_path}, {utterance.begin} - {utterance.end}"
                            )
                            traceback_lines = traceback.format_exception(
                                exc_type, exc_value, exc_traceback
                            )
                            extraction_logger.debug("\n".join(traceback_lines))
                            raise AlignmentCollectionError(
                                sound_file_path,
                                text_file_path,
                                utterance.begin,
                                utterance.end,
                                traceback_lines,
                                self.log_path,
                            )
                    alignment_archive.close()
                    extraction_logger.debug("Finished ali second pass")
                    ali_path = job.construct_path(
                        self.working_directory, "ali_first_pass", "ark", d.id
                    )
                    if ali_path.exists():
                        words_path = job.construct_path(
                            self.working_directory, "words_first_pass", "ark", d.id
                        )
                        likes_path = job.construct_path(
                            self.working_directory, "likelihoods_first_pass", "ark", d.id
                        )
                        alignment_archive = AlignmentArchive(
                            ali_path, words_file_name=words_path, likelihood_file_name=likes_path
                        )
                        missing_utterances = (
                            session.query(Utterance.kaldi_id)
                            .join(Utterance.speaker)
                            .filter(
                                Utterance.job_id == self.job_name, Speaker.dictionary_id == d.id
                            )
                            .filter(Utterance.ignored == False)  # noqa
                            .filter(~Utterance.id.in_(found_utterances))
                        )
                        for (utt_id,) in missing_utterances:
                            extraction_logger.debug(f"Processing {utt_id}")
                            try:
                                alignment = alignment_archive[utt_id]
                                intervals = alignment.generate_ctm(
                                    self.transition_model,
                                    lexicon_compiler.phone_table,
                                    self.frame_shift,
                                )
                                utterance = int(alignment.utterance_id.split("-")[-1])
                                try:
                                    ctm = lexicon_compiler.phones_to_pronunciations(
                                        alignment.words,
                                        intervals,
                                        transcription=False,
                                        text=utterance_texts.get(utterance, None),
                                    )
                                    ctm.update_utterance_boundaries(*utterance_times[utterance])
                                except Exception:
                                    exc_type, exc_value, exc_traceback = sys.exc_info()
                                    utterance, sound_file_path, text_file_path = (
                                        session.query(
                                            Utterance,
                                            SoundFile.sound_file_path,
                                            TextFile.text_file_path,
                                        )
                                        .join(Utterance.file)
                                        .join(File.sound_file)
                                        .join(File.text_file)
                                        .filter(Utterance.id == utterance)
                                        .first()
                                    )
                                    extraction_logger.debug(f"Error processing {utterance}:")
                                    extraction_logger.debug(
                                        f"Utterance information: {sound_file_path}, {text_file_path}, {utterance.begin} - {utterance.end}"
                                    )
                                    traceback_lines = traceback.format_exception(
                                        exc_type, exc_value, exc_traceback
                                    )
                                    extraction_logger.debug("\n".join(traceback_lines))
                                    raise AlignmentCollectionError(
                                        sound_file_path,
                                        text_file_path,
                                        utterance.begin,
                                        utterance.end,
                                        traceback_lines,
                                        self.log_path,
                                    )
                                self.callback((utterance, d.id, ctm))
                                extraction_logger.debug(f"Processed {utt_id}")
                            except (KeyError, RuntimeError):
                                extraction_logger.debug(f"Did not find {utt_id}")
                                pass
                        alignment_archive.close()
                        extraction_logger.debug("Finished ali first pass")
                del lexicon_compiler
            extraction_logger.debug("Finished extraction")


class ExportTextGridProcessWorker(mp.Process):
    """
    Multiprocessing worker for exporting TextGrids

    See Also
    --------
    :meth:`.CorpusAligner.collect_alignments`
        Main function that runs this worker in parallel

    Parameters
    ----------
    for_write_queue: :class:`~multiprocessing.Queue`
        Input queue of files to export
    stopped: :class:`~multiprocessing.Event`
        Stop check for processing
    finished_adding: :class:`~multiprocessing.Event`
        Input signal that all jobs have been added and no more new ones will come in
    textgrid_errors: dict[str, str]
        Dictionary for storing errors encountered
    arguments: :class:`~montreal_forced_aligner.alignment.multiprocessing.ExportTextGridArguments`
        Arguments to pass to the TextGrid export function
    exported_file_count: :class:`~montreal_forced_aligner.utils.Counter`
        Counter for exported files
    """

    def __init__(
        self,
        db_string: str,
        for_write_queue: mp.Queue,
        return_queue: mp.Queue,
        stopped: mp.Event,
        finished_adding: mp.Event,
        export_frame_shift: float,
        cleanup_textgrids: bool,
        clitic_marker: str,
        output_directory: Path,
        output_format: str,
        include_original_text: bool,
    ):
        super().__init__()
        self.db_string = db_string
        self.for_write_queue = for_write_queue
        self.return_queue = return_queue
        self.stopped = stopped
        self.finished_adding = finished_adding
        self.finished_processing = mp.Event()

        self.output_directory = output_directory
        self.output_format = output_format
        self.export_frame_shift = export_frame_shift
        self.include_original_text = include_original_text
        self.cleanup_textgrids = cleanup_textgrids
        self.clitic_marker = clitic_marker

    def run(self) -> None:
        """Run the exporter function"""
        db_engine = sqlalchemy.create_engine(self.db_string)
        with sqlalchemy.orm.Session(db_engine) as session:
            workflow: CorpusWorkflow = (
                session.query(CorpusWorkflow)
                .filter(CorpusWorkflow.current == True)  # noqa
                .first()
            )
            while True:
                try:
                    (file_batch) = self.for_write_queue.get(timeout=1)
                except Empty:
                    if self.finished_adding.is_set():
                        self.finished_processing.set()
                        break
                    continue

                if self.stopped.is_set():
                    continue
                try:
                    for output_path in construct_textgrid_output(
                        session,
                        file_batch,
                        workflow,
                        self.cleanup_textgrids,
                        self.clitic_marker,
                        self.output_directory,
                        self.export_frame_shift,
                        self.output_format,
                        self.include_original_text,
                    ):
                        self.return_queue.put(1)
                except Exception:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    self.return_queue.put(
                        AlignmentExportError(
                            output_path,
                            traceback.format_exception(exc_type, exc_value, exc_traceback),
                        )
                    )
                    self.stopped.set()
