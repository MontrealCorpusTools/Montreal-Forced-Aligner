"""
Alignment multiprocessing functions
-----------------------------------
"""
from __future__ import annotations

import collections
import json
import logging
import os
import statistics
import sys
import threading
import time
import traceback
import typing
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING, List

import numpy as np
import sqlalchemy
from _kalpy import feat as kalpy_feat
from _kalpy import transform as kalpy_transform
from _kalpy.gmm import gmm_compute_likes
from _kalpy.hmm import TransitionModel
from _kalpy.matrix import FloatMatrix, FloatSubMatrix
from _kalpy.util import RandomAccessBaseDoubleMatrixReader, RandomAccessBaseFloatMatrixReader
from kalpy.data import KaldiMapping, Segment
from kalpy.decoder.data import FstArchive
from kalpy.decoder.training_graphs import TrainingGraphCompiler
from kalpy.feat.data import FeatureArchive
from kalpy.feat.mfcc import MfccComputer
from kalpy.feat.pitch import PitchComputer
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.gmm.align import GmmAligner
from kalpy.gmm.data import AlignmentArchive, TranscriptionArchive
from kalpy.gmm.train import GmmStatsAccumulator
from kalpy.gmm.utils import read_gmm_model
from kalpy.utils import generate_read_specifier, kalpy_logger, read_kaldi_object
from sqlalchemy.orm import joinedload, selectinload, subqueryload

from montreal_forced_aligner.abc import KaldiFunction
from montreal_forced_aligner.data import (
    WORD_BEGIN_SYMBOL,
    WORD_END_SYMBOL,
    CtmInterval,
    MfaArguments,
    PhoneType,
    PronunciationProbabilityCounter,
    WordType,
)
from montreal_forced_aligner.db import (
    CorpusWorkflow,
    File,
    Job,
    Phone,
    PhoneInterval,
    SoundFile,
    Speaker,
    Utterance,
    Word,
)
from montreal_forced_aligner.exceptions import AlignmentExportError
from montreal_forced_aligner.helper import mfa_open, split_phone_position
from montreal_forced_aligner.textgrid import (
    construct_output_path,
    construct_output_tiers,
    export_textgrid,
)
from montreal_forced_aligner.utils import Counter, thread_logger

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
    "AnalyzeAlignmentsFunction",
    "AlignArguments",
    "AnalyzeAlignmentsArguments",
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
    db_string: str
        String for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
    text_int_paths: dict[int, Path]
        Per dictionary text SCP paths
    ali_paths: dict[int, Path]
        Per dictionary alignment paths
    model_path: :class:`~pathlib.Path`
        Acoustic model path
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
    db_string: str
        String for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
    model_path: :class:`~pathlib.Path`
        Acoustic model path
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
    db_string: str
        String for database connections
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
    db_string: str
        String for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
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
    db_string: str
        String for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
    model_path: :class:`~pathlib.Path`
        Path to model file
    align_options: dict[str, Any]
        Alignment options
    feature_options: dict[str, Any]
        Feature options
    confidence: bool
        Flag for outputting confidence
    """

    working_directory: Path
    aligner: GmmAligner
    feature_options: MetaDict
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
    db_string: str
        String for database connections
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
    db_string: str
        String for database connections
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
    db_string: str
        String for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
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
    db_string: str
        String for database connections
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
            if self.use_g2p:
                text_column = Utterance.normalized_character_text
            else:
                text_column = Utterance.normalized_text
            for d in job.dictionaries:
                begin = time.time()
                if self.lexicon_compilers and d.id in self.lexicon_compilers:
                    word_table = self.lexicon_compilers[d.id].word_table
                    lexicon = self.lexicon_compilers[d.id]
                else:
                    word_table = d.words_symbol_path
                    lexicon = d.lexicon_fst_path
                compiler = TrainingGraphCompiler(
                    self.model_path, self.tree_path, lexicon, word_table, use_g2p=self.use_g2p
                )
                graph_logger.debug(
                    f"Thread {self.job_name}: Set up took {time.time() - begin} seconds"
                )
                query = (
                    session.query(Utterance.kaldi_id, text_column)
                    .join(Utterance.speaker)
                    .filter(Utterance.job_id == self.job_name, Speaker.dictionary_id == d.id)
                    .filter(Utterance.ignored == False)  # noqa
                    .order_by(Utterance.kaldi_id)
                )
                if job.corpus.current_subset > 0:
                    query = query.filter(Utterance.in_subset == True)  # noqa
                graph_logger.info(f"Thread {self.job_name}: Compiling graphs for {d}")
                fst_ark_path = job.construct_path(workflow.working_directory, "fsts", "ark", d.id)
                total_time = 0
                compiler.export_graphs(
                    fst_ark_path,
                    query,
                    # callback=self.callback
                )
                graph_logger.debug(
                    f"Thread {self.job_name}: Total compilation time: {total_time} seconds"
                )
                del compiler
        del self.lexicon_compilers
        del self.session


def acc_stats_function(args: AccStatsArguments, lock: threading.Lock, transition_accs, gmm_accs):
    with args.session() as session, thread_logger(
        "kalpy.train", args.log_path, job_name=args.job_name
    ) as train_logger:
        job: Job = (
            session.query(Job)
            .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
            .filter(Job.id == args.job_name)
            .first()
        )
        for d in job.dictionaries:
            train_logger.debug(f"Accumulating stats for dictionary {d.id}")
            train_logger.debug(f"Accumulating stats for model: {args.model_path}")
            dict_id = d.id
            accumulator = GmmStatsAccumulator(args.model_path)

            fmllr_path = job.construct_path(
                job.corpus.current_subset_directory, "trans", "scp", dict_id
            )
            if not fmllr_path.exists():
                fmllr_path = None
            lda_mat_path = args.working_directory.joinpath("lda.mat")
            if not lda_mat_path.exists():
                lda_mat_path = None
            feat_path = job.construct_path(
                job.corpus.current_subset_directory, "feats", "scp", dictionary_id=dict_id
            )
            utt2spk_path = job.construct_path(
                job.corpus.current_subset_directory, "utt2spk", "scp", dict_id
            )
            utt2spk = KaldiMapping()
            utt2spk.load(utt2spk_path)
            train_logger.debug(f"Feature path: {feat_path}")
            train_logger.debug(f"LDA transform path: {lda_mat_path}")
            train_logger.debug(f"Speaker transform path: {fmllr_path}")
            train_logger.debug(f"utt2spk path: {utt2spk_path}")
            feature_archive = FeatureArchive(
                feat_path,
                utt2spk=utt2spk,
                lda_mat_file_name=lda_mat_path,
                transform_file_name=fmllr_path,
                deltas=True,
            )
            ali_path = job.construct_path(args.working_directory, "ali", "ark", dict_id)
            alignment_archive = AlignmentArchive(ali_path)
            accumulator.accumulate_stats(feature_archive, alignment_archive)
            with lock:
                transition_accs.AddVec(1.0, accumulator.transition_accs)
                gmm_accs.Add(1.0, accumulator.gmm_accs)


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

    def _run(self) -> typing.Generator[typing.Tuple[int, int]]:
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
            for d in job.dictionaries:
                train_logger.debug(f"Accumulating stats for dictionary {d.id}")
                train_logger.debug(f"Accumulating stats for model: {self.model_path}")
                dict_id = d.id
                accumulator = GmmStatsAccumulator(self.model_path)

                fmllr_path = job.construct_path(
                    job.corpus.current_subset_directory, "trans", "scp", dict_id
                )
                if not fmllr_path.exists():
                    fmllr_path = None
                lda_mat_path = self.working_directory.joinpath("lda.mat")
                if not lda_mat_path.exists():
                    lda_mat_path = None
                feat_path = job.construct_path(
                    job.corpus.current_subset_directory, "feats", "scp", dictionary_id=dict_id
                )
                utt2spk_path = job.construct_path(
                    job.corpus.current_subset_directory, "utt2spk", "scp", dict_id
                )
                utt2spk = KaldiMapping()
                utt2spk.load(utt2spk_path)
                train_logger.debug(f"Feature path: {feat_path}")
                train_logger.debug(f"LDA transform path: {lda_mat_path}")
                train_logger.debug(f"Speaker transform path: {fmllr_path}")
                train_logger.debug(f"utt2spk path: {utt2spk_path}")
                feature_archive = FeatureArchive(
                    feat_path,
                    utt2spk=utt2spk,
                    lda_mat_file_name=lda_mat_path,
                    transform_file_name=fmllr_path,
                    deltas=True,
                )
                ali_path = job.construct_path(self.working_directory, "ali", "ark", dict_id)
                alignment_archive = AlignmentArchive(ali_path)
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
        self.aligner = args.aligner
        self.confidence = args.confidence
        self.final = args.final

    def _run(self) -> None:
        """Run the function"""
        with self.session() as session, thread_logger(
            "kalpy.align", self.log_path, job_name=self.job_name
        ) as align_logger:
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )
            for d in job.dictionaries:
                align_logger.debug(f"Thread {self.job_name}: Aligning for dictionary {d.id}")
                align_logger.debug(
                    f"Thread {self.job_name}: Aligning with model: {self.aligner.acoustic_model_path}"
                )
                dict_id = d.id
                fst_path = job.construct_path(self.working_directory, "fsts", "ark", dict_id)
                align_logger.debug(f"Thread {self.job_name}: Training graph archive: {fst_path}")
                feature_archive = job.construct_feature_archive(self.working_directory, dict_id)

                training_graph_archive = FstArchive(fst_path)
                ali_path = job.construct_path(self.working_directory, "ali", "ark", dict_id)

                words_path = job.construct_path(self.working_directory, "words", "ark", dict_id)
                likes_path = job.construct_path(
                    self.working_directory, "likelihoods", "ark", dict_id
                )
                ali_path.unlink(missing_ok=True)
                words_path.unlink(missing_ok=True)
                likes_path.unlink(missing_ok=True)
                if self.aligner.acoustic_model_path.endswith(".alimdl"):
                    ali_path = job.construct_path(
                        self.working_directory, "ali_first_pass", "ark", dict_id
                    )
                    words_path = job.construct_path(
                        self.working_directory, "words_first_pass", "ark", dict_id
                    )
                    likes_path = job.construct_path(
                        self.working_directory, "likelihoods_first_pass", "ark", dict_id
                    )
                self.aligner.export_alignments(
                    ali_path,
                    training_graph_archive,
                    feature_archive,
                    word_file_name=words_path,
                    likelihood_file_name=likes_path,
                    callback=self.callback,
                )
                if self.aligner.acoustic_model_path.endswith(".alimdl"):
                    job.construct_path(self.working_directory, "ali", "ark", dict_id).symlink_to(
                        ali_path
                    )
                    job.construct_path(self.working_directory, "words", "ark", dict_id).symlink_to(
                        words_path
                    )
                    job.construct_path(
                        self.working_directory, "likelihoods", "ark", dict_id
                    ).symlink_to(likes_path)


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
                    Phone.phone_type.in_([PhoneType.non_silence, PhoneType.oov]),
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
                duration_zscore_sum = 0
                for pi in phone_intervals:
                    log_like_sum += pi.phone_goodness
                    m, sd = phones[pi.phone_id]
                    duration_zscore_sum += abs((pi.duration - m) / sd)
                utterance_speech_log_likelihood = log_like_sum / interval_count
                utterance_duration_deviation = duration_zscore_sum / interval_count
                self.callback(
                    (utterance.id, utterance_speech_log_likelihood, utterance_duration_deviation)
                )


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
        self.feature_padding_factor = 4
        self.padding = round(self.frame_shift_seconds, 3)
        self.splice_frames = 3

    def _run(self) -> typing.Generator[typing.Tuple[int, float]]:
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
                    Phone.phone_type.in_([PhoneType.silence, PhoneType.oov])
                )
            ]

            cmvn_paths = job.per_dictionary_cmvn_scp_paths
            trans_paths = job.per_dictionary_trans_scp_paths
            compiler = TrainingGraphCompiler(
                self.model_path,
                self.tree_path,
                self.lexicon_compiler,
                self.lexicon_compiler.word_table,
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
                        segment_begin = max(round(interval.begin - self.padding, 4), 0)
                        feature_segment_begin = max(
                            round(
                                interval.begin - (self.padding * self.feature_padding_factor), 4
                            ),
                            0,
                        )
                        segment_end = min(round(interval.begin + self.padding, 4), utterance.end)
                        feature_segment_end = min(
                            round(
                                interval.begin + (self.padding * self.feature_padding_factor), 4
                            ),
                            utterance.end,
                        )
                        begin_offset = round(segment_begin - feature_segment_begin, 4)
                        end_offset = round(segment_end - feature_segment_begin, 4)
                        segment = Segment(
                            sf_path, feature_segment_begin, feature_segment_end, utterance.channel
                        )
                        text = f"{self.phone_to_group_mapping[prev_label]} {self.phone_to_group_mapping[phone]}"

                        train_graph = compiler.compile_fst(text)

                        prev_label = phone
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
                        ctm_intervals = alignment.generate_ctm(
                            aligner.transition_model, self.lexicon_compiler.phone_table
                        )
                        interval_mapping.append(
                            {
                                "id": interval.id,
                                "begin": round(ctm_intervals[1].begin + feature_segment_begin, 4),
                                "end": interval.end,
                                "label": phone_mapping[ctm_intervals[1].label],
                            }
                        )
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
                        interval_mappings.append({"id": pi.id, "phone_goodness": average_score})
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

    def _run(self) -> typing.Generator[typing.Tuple[int, int, str]]:
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

            for d in job.dictionaries:
                ali_path = job.construct_path(workflow.working_directory, "ali", "ark", d.id)
                if not os.path.exists(ali_path):
                    continue

                utts = (
                    session.query(Utterance.id, Utterance.normalized_text)
                    .join(Utterance.speaker)
                    .filter(Utterance.job_id == self.job_name)
                    .filter(Speaker.dictionary_id == d.id)
                )
                utterance_texts = {}
                for u_id, text in utts:
                    utterance_texts[u_id] = text
                lexicon_compiler = self.lexicon_compilers[d.id]

                words_path = job.construct_path(workflow.working_directory, "words", "ark", d.id)
                alignment_archive = AlignmentArchive(ali_path, words_file_name=words_path)
                for alignment in alignment_archive:
                    intervals = alignment.generate_ctm(
                        self.aligner.transition_model, lexicon_compiler.phone_table
                    )
                    utterance = int(alignment.utterance_id.split("-")[-1])
                    ctm = lexicon_compiler.phones_to_pronunciations(
                        utterance_texts[utterance], alignment.words, intervals
                    )
                    word_pronunciations = [(x.label, x.pronunciation) for x in ctm.word_intervals]
                    # word_pronunciations = [
                    #    x if x[1] != OOV_PHONE else (OOV_WORD, OOV_PHONE)
                    #    for x in word_pronunciations
                    # ]
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

    def _run(self) -> typing.Generator[typing.Tuple[int, List[CtmInterval], List[CtmInterval]]]:
        """Run the function"""
        with self.session() as session, kalpy_logger("kalpy.align", self.log_path):
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
                columns = [Utterance.id, Utterance.begin, Utterance.end]
                if d.use_g2p:
                    columns.append(Utterance.normalized_character_text)
                else:
                    columns.append(Utterance.normalized_text)
                utts = (
                    session.query(*columns)
                    .join(Utterance.speaker)
                    .filter(Utterance.job_id == self.job_name)
                    .filter(Speaker.dictionary_id == d.id)
                )
                utterance_begins = {}
                utterance_ends = {}
                utterance_texts = {}
                for u_id, begin, end, text in utts:
                    utterance_begins[u_id] = begin
                    utterance_ends[u_id] = end
                    utterance_texts[u_id] = text
                lexicon_compiler = self.lexicon_compilers[d.id]
                if self.transcription:
                    lat_path = job.construct_path(workflow.working_directory, "lat", "ark", d.id)

                    transcription_archive = TranscriptionArchive(
                        lat_path, acoustic_scale=self.score_options["acoustic_scale"]
                    )
                    for transcription in transcription_archive:
                        intervals = transcription.generate_ctm(
                            self.transition_model, lexicon_compiler.phone_table, self.frame_shift
                        )
                        utterance = int(transcription.utterance_id.split("-")[-1])
                        ctm = lexicon_compiler.phones_to_pronunciations(
                            utterance_texts[utterance],
                            transcription.words,
                            intervals,
                            transcription=True,
                        )
                        ctm.update_utterance_boundaries(
                            utterance_begins[utterance], utterance_ends[utterance]
                        )
                        self.callback((utterance, d.id, ctm))
                else:
                    ali_path = job.construct_path(workflow.working_directory, "ali", "ark", d.id)
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
                        ctm = lexicon_compiler.phones_to_pronunciations(
                            utterance_texts[utterance],
                            alignment.words,
                            intervals,
                            transcription=False,
                        )
                        ctm.update_utterance_boundaries(
                            utterance_begins[utterance], utterance_ends[utterance]
                        )
                        self.callback((utterance, d.id, ctm))
                    alignment_archive.close()

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
                            try:
                                alignment = alignment_archive[utt_id]
                                intervals = alignment.generate_ctm(
                                    self.transition_model,
                                    lexicon_compiler.phone_table,
                                    self.frame_shift,
                                )
                                utterance = int(alignment.utterance_id.split("-")[-1])
                                found_utterances.add(utterance)
                                ctm = lexicon_compiler.phones_to_pronunciations(
                                    utterance_texts[utterance],
                                    alignment.words,
                                    intervals,
                                    transcription=False,
                                )
                                ctm.update_utterance_boundaries(
                                    utterance_begins[utterance], utterance_ends[utterance]
                                )
                                self.callback((utterance, d.id, ctm))
                            except (KeyError, RuntimeError):
                                pass


class ExportTextGridProcessWorker(threading.Thread):
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
    stopped: :class:`~threading.Event`
        Stop check for processing
    finished_processing: :class:`~threading.Event`
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
        session: sqlalchemy.orm.scoped_session,
        for_write_queue: Queue,
        return_queue: Queue,
        stopped: threading.Event,
        finished_adding: threading.Event,
        arguments: ExportTextGridArguments,
        exported_file_count: Counter,
    ):
        super().__init__()
        self.session = session
        self.for_write_queue = for_write_queue
        self.return_queue = return_queue
        self.stopped = stopped
        self.finished_adding = finished_adding
        self.finished_processing = threading.Event()

        self.output_directory = arguments.output_directory
        self.output_format = arguments.output_format
        self.export_frame_shift = arguments.export_frame_shift
        self.log_path = arguments.log_path
        self.include_original_text = arguments.include_original_text
        self.cleanup_textgrids = arguments.cleanup_textgrids
        self.clitic_marker = arguments.clitic_marker
        self.exported_file_count = exported_file_count

    def run(self) -> None:
        """Run the exporter function"""
        with mfa_open(self.log_path, "w") as log_file, self.session() as session:
            workflow: CorpusWorkflow = (
                session.query(CorpusWorkflow)
                .filter(CorpusWorkflow.current == True)  # noqa
                .first()
            )
            log_file.write(f"Exporting TextGrids for Workflow ID: {workflow.id}\n")
            log_file.write(f"Output directory: {self.output_directory}\n")
            log_file.write(f"Output format: {self.output_format}\n")
            log_file.write(f"Frame shift: {self.export_frame_shift}\n")
            log_file.write(f"Include original text: {self.include_original_text}\n")
            log_file.write(f"Clean up textgrids: {self.cleanup_textgrids}\n")
            while True:
                try:
                    (
                        file_id,
                        name,
                        relative_path,
                        duration,
                        text_file_path,
                    ) = self.for_write_queue.get(timeout=1)
                except Empty:
                    if self.finished_adding.is_set():
                        self.finished_processing.set()
                        break
                    continue

                if self.stopped.is_set():
                    continue
                try:
                    output_path = construct_output_path(
                        name,
                        relative_path,
                        self.output_directory,
                        text_file_path,
                        self.output_format,
                    )
                    data = construct_output_tiers(
                        session,
                        file_id,
                        workflow,
                        self.cleanup_textgrids,
                        self.clitic_marker,
                        self.include_original_text,
                    )
                    export_textgrid(
                        data, output_path, duration, self.export_frame_shift, self.output_format
                    )
                    self.return_queue.put(1)
                    self.for_write_queue.task_done()
                except Exception:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    self.return_queue.put(
                        AlignmentExportError(
                            output_path,
                            traceback.format_exception(exc_type, exc_value, exc_traceback),
                        )
                    )
                    self.stopped.set()
            log_file.write("Done!\n")
