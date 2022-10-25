"""
Alignment multiprocessing functions
-----------------------------------

"""
from __future__ import annotations

import collections
import json
import multiprocessing as mp
import os
import re
import statistics
import subprocess
import sys
import traceback
import typing
from queue import Empty
from typing import TYPE_CHECKING, Dict, List, Union

import numpy as np
import pynini
import pywrapfst
import sqlalchemy.engine
from sqlalchemy.orm import Session, joinedload, load_only, selectinload

from montreal_forced_aligner.corpus.features import (
    compute_feature_process,
    compute_transform_process,
)
from montreal_forced_aligner.data import (
    CtmInterval,
    MfaArguments,
    PronunciationProbabilityCounter,
    TextgridFormats,
    WordCtmInterval,
    WordType,
    WorkflowType,
)
from montreal_forced_aligner.db import (
    CorpusWorkflow,
    DictBundle,
    Dictionary,
    File,
    Phone,
    PhoneInterval,
    Pronunciation,
    SoundFile,
    Speaker,
    Utterance,
    Word,
    WordInterval,
)
from montreal_forced_aligner.exceptions import AlignmentExportError
from montreal_forced_aligner.helper import mfa_open, split_phone_position
from montreal_forced_aligner.textgrid import export_textgrid
from montreal_forced_aligner.utils import (
    Counter,
    KaldiFunction,
    Stopped,
    parse_ctm_output,
    read_feats,
    thirdparty_binary,
)

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
    "AccStatsFunction",
    "AccStatsArguments",
    "compile_information_func",
    "CompileInformationArguments",
    "CompileTrainGraphsFunction",
    "CompileTrainGraphsArguments",
    "GeneratePronunciationsArguments",
    "GeneratePronunciationsFunction",
]


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
    log_path: str
        Path to save logging information during the run
    text_int_paths: dict[int, str]
        Per dictionary text SCP paths
    ali_paths: dict[int, str]
        Per dictionary alignment paths
    model_path: str
        Acoustic model path
    for_g2p: bool
        Flag for training a G2P model with acoustic information
    """

    text_int_paths: Dict[int, str]
    ali_paths: Dict[int, str]
    model_path: str
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
    log_path: str
        Path to save logging information during the run
    model_path: str
        Acoustic model path
    frame_shift: float
        Frame shift in seconds
    ali_paths: dict[int, str]
        Per dictionary alignment paths
    text_int_paths: dict[int, str]
        Per dictionary text SCP paths
    phone_symbol_path: str
        Path to phone symbols table
    score_options: dict[str, Any]
        Options for Kaldi functions
    """

    model_path: str
    frame_shift: float
    ali_paths: Dict[int, str]
    text_int_paths: Dict[int, str]
    phone_symbol_path: str
    score_options: MetaDict


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
    log_path: str
        Path to save logging information during the run
    export_frame_shift: float
        Frame shift in seconds
    cleanup_textgrids: bool
        Flag to cleanup silences and recombine words
    clitic_marker: str
        Marker indicating clitics
    output_directory: str
        Directory for exporting
    output_format: str
        Format to export
    include_original_text: bool
        Flag for including original unnormalized text as a tier
    workflow_id: int
        Integer id of workflow to export
    """

    export_frame_shift: float
    cleanup_textgrids: bool
    clitic_marker: str
    output_directory: str
    output_format: str
    include_original_text: bool
    workflow_id: int


@dataclass
class CompileInformationArguments(MfaArguments):
    """
    Arguments for :func:`~montreal_forced_aligner.alignment.multiprocessing.compile_information_func`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    db_string: str
        String for database connections
    log_path: str
        Path to save logging information during the run
    align_log_path: str
        Path to log file for parsing
    """

    align_log_path: str


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
    log_path: str
        Path to save logging information during the run
    dictionaries: list[int]
        List of dictionary ids
    tree_path: str
        Path to tree file
    model_path: str
        Path to model file
    text_int_paths: dict[int, str]
        Mapping of dictionaries to text scp files
    fst_ark_paths: dict[int, str]
        Mapping of dictionaries to fst ark files
    """

    dictionaries: List[int]
    tree_path: str
    model_path: str
    text_int_paths: Dict[int, str]
    fst_ark_paths: Dict[int, str]


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
    log_path: str
        Path to save logging information during the run
    dictionaries: list[int]
        List of dictionary ids
    fst_ark_paths: dict[int, str]
        Mapping of dictionaries to fst ark files
    feature_strings: dict[int, str]
        Mapping of dictionaries to feature generation strings
    model_path: str
        Path to model file
    ali_paths: dict[int, str]
        Per dictionary alignment paths
    align_options: dict[str, Any]
        Alignment options
    """

    dictionaries: List[int]
    fst_ark_paths: Dict[int, str]
    feature_strings: Dict[int, str]
    model_path: str
    ali_paths: Dict[int, str]
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
    log_path: str
        Path to save logging information during the run
    working_directory: str
        Current working directory
    tree_path: str
        Path to tree file
    model_path: str
        Path to model file
    frame_shift: int
        Frame shift in ms
    cmvn_paths: dict[int, str]
        Mapping of dictionaries to CMVN scp paths
    fmllr_paths: dict[int, str]
        Mapping of dictionaries to fMLLR ark paths
    lda_mat_path: str, optional
        Path to LDA matrix file
    mfcc_options: dict[str, Any]
        MFCC computation options
    pitch_options: dict[str, Any]
        Pitch computation options
    align_options: dict[str, Any]
        Alignment options
    workflow_id: int
        Integer ID for workflow to fine tune
    position_dependent_phones: bool
        Flag for whether to use position dependent phones
    grouped_phones: dict[str, list[str]]
        Grouped lists of phones
    """

    working_directory: str
    phone_symbol_table_path: str
    disambiguation_symbols_int_path: str
    tree_path: str
    model_path: str
    frame_shift: int
    cmvn_paths: Dict[int, str]
    fmllr_paths: Dict[int, str]
    lda_mat_path: typing.Optional[str]
    mfcc_options: MetaDict
    pitch_options: MetaDict
    lda_options: MetaDict
    align_options: MetaDict
    workflow_id: int
    position_dependent_phones: bool
    grouped_phones: Dict[str, List[str]]


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
    log_path: str
        Path to save logging information during the run
    model_path: str
        Path to model file
    phone_pdf_counts_path: str
        Path to output PDF counts
    feature_strings: dict[int, str]
        Mapping of dictionaries to feature generation strings
    """

    model_path: str
    phone_pdf_counts_path: str
    feature_strings: Dict[int, str]


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
    log_path: str
        Path to save logging information during the run
    dictionaries: list[int]
        List of dictionary ids
    feature_strings: dict[int, str]
        Mapping of dictionaries to feature generation strings
    ali_paths: dict[int, str]
        Per dictionary alignment paths
    acc_paths: dict[int, str]
        Per dictionary accumulated stats paths
    model_path: str
        Path to model file
    """

    dictionaries: List[int]
    feature_strings: Dict[int, str]
    ali_paths: Dict[int, str]
    acc_paths: Dict[int, str]
    model_path: str


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

    progress_pattern = re.compile(
        r"^LOG.*succeeded for (?P<succeeded>\d+) graphs, failed for (?P<failed>\d+)"
    )

    def __init__(self, args: CompileTrainGraphsArguments):
        super().__init__(args)
        self.dictionaries = args.dictionaries
        self.tree_path = args.tree_path
        self.model_path = args.model_path
        self.text_int_paths = args.text_int_paths
        self.fst_ark_paths = args.fst_ark_paths
        self.working_dir = os.path.dirname(list(self.fst_ark_paths.values())[0])

    def _run(self) -> typing.Generator[typing.Tuple[int, int]]:
        """Run the function"""
        db_engine = sqlalchemy.create_engine(self.db_string)

        with mfa_open(self.log_path, "w") as log_file, Session(db_engine) as session:
            dictionaries = (
                session.query(Dictionary)
                .join(Dictionary.speakers)
                .join(Speaker.utterances)
                .filter(Utterance.job_id == self.job_name)
                .distinct()
            )

            tree_proc = subprocess.Popen(
                [thirdparty_binary("tree-info"), self.tree_path],
                encoding="utf8",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, _ = tree_proc.communicate()
            context_width = 1
            central_pos = 0
            for line in stdout.split("\n"):
                text = line.strip().split(" ")
                if text[0] == "context-width":
                    context_width = int(text[1])
                elif text[0] == "central-position":
                    central_pos = int(text[1])
            out_disambig = os.path.join(self.working_dir, f"{self.job_name}.disambig")
            ilabels_temp = os.path.join(self.working_dir, f"{self.job_name}.ilabels")
            clg_path = os.path.join(self.working_dir, f"{self.job_name}.clg.temp")
            ha_out_disambig = os.path.join(
                self.working_dir, f"{self.job_name}.ha_out_disambig.temp"
            )
            for d in dictionaries:
                fst_ark_path = self.fst_ark_paths[d.id]
                text_path = self.text_int_paths[d.id]
                if d.use_g2p:
                    import pynini
                    from pynini.lib import rewrite

                    from montreal_forced_aligner.g2p.generator import threshold_lattice_to_dfa

                    fst = pynini.Fst.read(d.lexicon_fst_path)
                    token_type = pynini.SymbolTable.read_text(d.grapheme_symbol_table_path)
                    utterances = (
                        session.query(Utterance.kaldi_id, Utterance.normalized_character_text)
                        .join(Utterance.speaker)
                        .filter(Utterance.ignored == False)  # noqa
                        .filter(Utterance.normalized_character_text != "")
                        .filter(Utterance.job_id == self.job_name)
                        .filter(Speaker.dictionary_id == d.id)
                        .order_by(Utterance.kaldi_id)
                    )
                    with mfa_open(fst_ark_path, "wb") as fst_output_file:
                        for utt_id, full_text in utterances:
                            full_text = f"<s> {full_text} </s>"
                            lattice = rewrite.rewrite_lattice(full_text, fst, token_type)
                            lattice = threshold_lattice_to_dfa(lattice, 2.0)
                            input = lattice.write_to_string()
                            clg_compose_proc = subprocess.Popen(
                                [
                                    thirdparty_binary("fstcomposecontext"),
                                    f"--context-size={context_width}",
                                    f"--central-position={central_pos}",
                                    f"--read-disambig-syms={d.disambiguation_symbols_int_path}",
                                    f"--write-disambig-syms={out_disambig}",
                                    ilabels_temp,
                                    "-",
                                    "-",
                                ],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=log_file,
                                env=os.environ,
                            )
                            clg_sort_proc = subprocess.Popen(
                                [
                                    thirdparty_binary("fstarcsort"),
                                    "--sort_type=ilabel",
                                    "-",
                                    clg_path,
                                ],
                                stdin=clg_compose_proc.stdout,
                                stderr=log_file,
                                env=os.environ,
                            )
                            clg_compose_proc.stdin.write(input)
                            clg_compose_proc.stdin.flush()
                            clg_compose_proc.stdin.close()
                            clg_sort_proc.communicate()

                            make_h_proc = subprocess.Popen(
                                [
                                    thirdparty_binary("make-h-transducer"),
                                    f"--disambig-syms-out={ha_out_disambig}",
                                    ilabels_temp,
                                    self.tree_path,
                                    self.model_path,
                                ],
                                stderr=log_file,
                                stdout=subprocess.PIPE,
                                env=os.environ,
                            )
                            hclg_compose_proc = subprocess.Popen(
                                [thirdparty_binary("fsttablecompose"), "-", clg_path, "-"],
                                stderr=log_file,
                                stdin=make_h_proc.stdout,
                                stdout=subprocess.PIPE,
                                env=os.environ,
                            )

                            hclg_determinize_proc = subprocess.Popen(
                                [thirdparty_binary("fstdeterminizestar"), "--use-log=true"],
                                stdin=hclg_compose_proc.stdout,
                                stdout=subprocess.PIPE,
                                stderr=log_file,
                                env=os.environ,
                            )
                            hclg_rmsymbols_proc = subprocess.Popen(
                                [thirdparty_binary("fstrmsymbols"), ha_out_disambig],
                                stdin=hclg_determinize_proc.stdout,
                                stdout=subprocess.PIPE,
                                stderr=log_file,
                                env=os.environ,
                            )
                            hclg_rmeps_proc = subprocess.Popen(
                                [thirdparty_binary("fstrmepslocal")],
                                stdin=hclg_rmsymbols_proc.stdout,
                                stdout=subprocess.PIPE,
                                stderr=log_file,
                                env=os.environ,
                            )
                            hclg_minimize_proc = subprocess.Popen(
                                [thirdparty_binary("fstminimizeencoded")],
                                stdin=hclg_rmeps_proc.stdout,
                                stdout=subprocess.PIPE,
                                stderr=log_file,
                                env=os.environ,
                            )
                            hclg_self_loop_proc = subprocess.Popen(
                                [
                                    thirdparty_binary("add-self-loops"),
                                    "--self-loop-scale=0.1",
                                    "--reorder=true",
                                    self.model_path,
                                    "-",
                                    "-",
                                ],
                                stdin=hclg_minimize_proc.stdout,
                                stdout=subprocess.PIPE,
                                stderr=log_file,
                                env=os.environ,
                            )

                            stdout, _ = hclg_self_loop_proc.communicate()
                            self.check_call(hclg_minimize_proc)
                            fst_output_file.write(utt_id.encode("utf8") + b" ")
                            fst_output_file.write(stdout)
                            yield 1, 0

                else:
                    proc = subprocess.Popen(
                        [
                            thirdparty_binary("compile-train-graphs"),
                            f"--read-disambig-syms={d.disambiguation_symbols_int_path}",
                            self.tree_path,
                            self.model_path,
                            d.lexicon_fst_path,
                            f"ark:{text_path}",
                            f"ark:{fst_ark_path}",
                        ],
                        stderr=subprocess.PIPE,
                        encoding="utf8",
                        env=os.environ,
                    )
                    for line in proc.stderr:
                        log_file.write(line)
                        m = self.progress_pattern.match(line.strip())
                        if m:
                            yield int(m.group("succeeded")), int(m.group("failed"))
                    self.check_call(proc)


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

    progress_pattern = re.compile(
        r"^LOG \(gmm-acc-stats-ali.* Processed (?P<utterances>\d+) utterances;.*"
    )

    done_pattern = re.compile(
        r"^LOG \(gmm-acc-stats-ali.*Done (?P<utterances>\d+) files, (?P<errors>\d+) with errors.$"
    )

    def __init__(self, args: AccStatsArguments):
        super().__init__(args)
        self.dictionaries = args.dictionaries
        self.feature_strings = args.feature_strings
        self.model_path = args.model_path
        self.ali_paths = args.ali_paths
        self.acc_paths = args.acc_paths

    def _run(self) -> typing.Generator[typing.Tuple[int, int]]:
        """Run the function"""
        with mfa_open(self.log_path, "w") as log_file:
            for dict_id in self.dictionaries:
                processed_count = 0
                acc_proc = subprocess.Popen(
                    [
                        thirdparty_binary("gmm-acc-stats-ali"),
                        self.model_path,
                        self.feature_strings[dict_id],
                        f"ark,s,cs:{self.ali_paths[dict_id]}",
                        self.acc_paths[dict_id],
                    ],
                    stderr=subprocess.PIPE,
                    encoding="utf8",
                    env=os.environ,
                )
                for line in acc_proc.stderr:
                    log_file.write(line)
                    m = self.progress_pattern.match(line.strip())
                    if m:
                        now_processed = int(m.group("utterances"))
                        progress_update = now_processed - processed_count
                        processed_count = now_processed
                        yield progress_update, 0
                    else:
                        m = self.done_pattern.match(line.strip())
                        if m:
                            now_processed = int(m.group("utterances"))
                            progress_update = now_processed - processed_count
                            yield progress_update, int(m.group("errors"))
                self.check_call(acc_proc)


class AlignFunction(KaldiFunction):
    """
    Multiprocessing function for alignment.

    See Also
    --------
    :meth:`.AlignMixin.align_utterances`
        Main function that calls this function in parallel
    :meth:`.AlignMixin.align_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`align-equal-compiled`
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
        self.dictionaries = args.dictionaries
        self.fst_ark_paths = args.fst_ark_paths
        self.feature_strings = args.feature_strings
        self.model_path = args.model_path
        self.ali_paths = args.ali_paths
        self.align_options = args.align_options

    def _run(self) -> typing.Generator[typing.Tuple[int, float]]:
        """Run the function"""
        with mfa_open(self.log_path, "w") as log_file:
            for dict_id in self.dictionaries:
                feature_string = self.feature_strings[dict_id]
                fst_path = self.fst_ark_paths[dict_id]
                ali_path = self.ali_paths[dict_id]
                com = [
                    thirdparty_binary("gmm-align-compiled"),
                    f"--transition-scale={self.align_options['transition_scale']}",
                    f"--acoustic-scale={self.align_options['acoustic_scale']}",
                    f"--self-loop-scale={self.align_options['self_loop_scale']}",
                    f"--beam={self.align_options['beam']}",
                    f"--retry-beam={self.align_options['retry_beam']}",
                    "--careful=false",
                    "-",
                    f"ark:{fst_path}",
                    feature_string,
                    f"ark:{ali_path}",
                    "ark,t:-",
                ]

                boost_proc = subprocess.Popen(
                    [
                        thirdparty_binary("gmm-boost-silence"),
                        f"--boost={self.align_options['boost_silence']}",
                        self.align_options["optional_silence_csl"],
                        self.model_path,
                        "-",
                    ],
                    stderr=log_file,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                )
                align_proc = subprocess.Popen(
                    com,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    encoding="utf8",
                    stdin=boost_proc.stdout,
                    env=os.environ,
                )
                for line in align_proc.stdout:
                    line = line.strip()
                    utterance, log_likelihood = line.split()
                    u_id = int(utterance.split("-")[-1])
                    yield u_id, float(log_likelihood)
                self.check_call(align_proc)


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
        self.frame_shift = args.frame_shift
        self.scaling_factor = 10

        self.frame_shift_seconds = round(self.frame_shift / 1000, 3)
        self.new_frame_shift = int(self.frame_shift / self.scaling_factor)
        self.new_frame_shift_seconds = round(self.new_frame_shift / 1000, 4)
        self.feature_padding_factor = 4
        self.padding = round(self.frame_shift_seconds, 3)
        self.working_directory = args.working_directory
        self.tree_path = args.tree_path
        self.model_path = args.model_path
        self.cmvn_paths = args.cmvn_paths
        self.fmllr_paths = args.fmllr_paths
        self.lda_mat_path = args.lda_mat_path
        self.mfcc_options = args.mfcc_options
        self.mfcc_options["frame-shift"] = self.new_frame_shift
        self.mfcc_options["snip-edges"] = False
        self.pitch_options = args.pitch_options
        self.pitch_options["frame-shift"] = self.new_frame_shift
        self.pitch_options["snip-edges"] = False
        self.lda_options = args.lda_options
        self.align_options = args.align_options
        self.workflow_id = args.workflow_id
        self.phone_symbol_table_path = args.phone_symbol_table_path
        self.position_dependent_phones = args.position_dependent_phones
        self.disambiguation_symbols_int_path = args.disambiguation_symbols_int_path
        self.grouped_phones = args.grouped_phones
        self.segment_begins = {}
        self.segment_ends = {}
        self.original_intervals = {}
        self.utterance_initial_intervals = {}

    def setup_files(self, session, dictionary_id, phone_mapping):
        wav_path = os.path.join(
            self.working_directory, f"fine_tune_wav.{dictionary_id}.{self.job_name}.scp"
        )
        segment_path = os.path.join(
            self.working_directory, f"fine_tune_segments.{dictionary_id}.{self.job_name}.scp"
        )
        feature_segment_path = os.path.join(
            self.working_directory,
            f"fine_tune_feature_segments.{dictionary_id}.{self.job_name}.scp",
        )
        utt2spk_path = os.path.join(
            self.working_directory, f"fine_tune_utt2spk.{dictionary_id}.{self.job_name}.scp"
        )
        text_path = os.path.join(
            self.working_directory, f"fine_tune_text.{dictionary_id}.{self.job_name}.scp"
        )

        columns = [
            PhoneInterval.utterance_id,
            Phone.kaldi_label,
            PhoneInterval.id,
            PhoneInterval.begin,
            PhoneInterval.end,
            SoundFile.sox_string,
            SoundFile.sound_file_path,
            SoundFile.sample_rate,
            Utterance.channel,
            Utterance.speaker_id,
            Utterance.file_id,
        ]
        utterance_ends = {
            k: v
            for k, v in session.query(Utterance.id, Utterance.end).filter(
                Utterance.job_id == self.job_name
            )
        }
        bn = DictBundle("interval_data", *columns)

        interval_query = (
            session.query(bn)
            .join(PhoneInterval.phone)
            .join(PhoneInterval.utterance)
            .join(Utterance.file)
            .join(File.sound_file)
            .filter(Utterance.job_id == self.job_name)
            .filter(PhoneInterval.workflow_id == self.workflow_id)
            .order_by(PhoneInterval.utterance_id, PhoneInterval.begin)
        )
        wav_data = {}
        utt2spk_data = {}
        segment_data = {}
        text_data = {}
        prev_label = None
        current_id = None
        for row in interval_query:
            data = row.interval_data
            if current_id is None:
                current_id = data["utterance_id"]
            label = data["kaldi_label"]
            if current_id != data["utterance_id"] or prev_label is None:
                self.utterance_initial_intervals[data["utterance_id"]] = {
                    "id": data["id"],
                    "begin": data["begin"],
                    "end": data["end"],
                }
                prev_label = label
                current_id = data["utterance_id"]
                continue
            boundary_id = f"{data['utterance_id']}-{data['id']}"
            utt2spk_data[boundary_id] = data["speaker_id"]
            sox_string = data["sox_string"]
            if not sox_string:
                sox_string = f'sox "{data["sound_file_path"]}" -t wav -b 16 -r 16000 - |'
            wav_data[str(data["file_id"])] = sox_string
            interval_begin = data["begin"]
            self.original_intervals[data["id"]] = {
                "begin": data["begin"],
                "end": data["end"],
                "utterance_id": data["utterance_id"],
            }
            segment_begin = round(interval_begin - self.padding, 4)
            feature_segment_begin = round(
                interval_begin - (self.padding * self.feature_padding_factor), 4
            )
            if segment_begin < 0:
                segment_begin = 0
            if feature_segment_begin < 0:
                feature_segment_begin = 0
            begin_offset = round(segment_begin - feature_segment_begin, 4)
            segment_end = round(interval_begin + self.padding, 4)
            feature_segment_end = round(
                interval_begin + (self.padding * self.feature_padding_factor), 4
            )
            if segment_end > utterance_ends[data["utterance_id"]]:
                segment_end = utterance_ends[data["utterance_id"]]
            if feature_segment_end > utterance_ends[data["utterance_id"]]:
                feature_segment_end = utterance_ends[data["utterance_id"]]
            end_offset = round(segment_end - feature_segment_begin, 4)
            self.segment_begins[data["id"]] = segment_begin
            self.segment_ends[data["id"]] = data["end"]
            segment_data[boundary_id] = (
                map(
                    str,
                    [
                        data["file_id"],
                        f"{feature_segment_begin:.4f}",
                        f"{feature_segment_end:.4f}",
                        data["channel"],
                    ],
                ),
                map(str, [boundary_id, f"{begin_offset:.4f}", f"{end_offset:.4f}"]),
            )

            text_data[
                boundary_id
            ] = f"{self.phone_to_group_mapping[prev_label]} {self.phone_to_group_mapping[label]}"
            prev_label = label

        with mfa_open(utt2spk_path, "w") as f:
            for k, v in sorted(utt2spk_data.items()):
                f.write(f"{k} {v}\n")
        with mfa_open(wav_path, "w") as f:
            for k, v in sorted(wav_data.items()):
                f.write(f"{k} {v}\n")
        with mfa_open(segment_path, "w") as f, mfa_open(feature_segment_path, "w") as feature_f:
            for k, v in sorted(segment_data.items()):
                f.write(f"{k} {' '.join(v[0])}\n")
                feature_f.write(f"{k} {' '.join(v[1])}\n")
        with mfa_open(text_path, "w") as f:
            for k, v in sorted(text_data.items()):
                f.write(f"{k} {v}\n")

    def _run(self) -> typing.Generator[typing.Tuple[int, float]]:
        """Run the function"""
        db_engine = sqlalchemy.create_engine(self.db_string)
        with Session(db_engine) as session, mfa_open(self.log_path, "w") as log_file:

            reversed_phone_mapping = {}
            phone_mapping = {}
            phone_query = session.query(Phone.mapping_id, Phone.id, Phone.kaldi_label)
            for m_id, p_id, phone in phone_query:
                reversed_phone_mapping[m_id] = p_id
                phone_mapping[phone] = m_id

            lexicon_path = os.path.join(self.working_directory, "phone.fst")
            group_mapping_path = os.path.join(self.working_directory, "groups.txt")
            fst = pynini.Fst()
            initial_state = fst.add_state()
            fst.set_start(initial_state)
            fst.set_final(initial_state, 0)
            processed = set()
            if self.position_dependent_phones:
                self.grouped_phones["silence"] = ["sil", "sil_B", "sil_I", "sil_E", "sil_S"]
                self.grouped_phones["unknown"] = ["spn", "spn_B", "spn_I", "spn_E", "spn_S"]
            else:
                self.grouped_phones["silence"] = ["sil"]
                self.grouped_phones["unknown"] = ["spn"]
            group_set = ["<eps>"] + sorted(k for k in self.grouped_phones.keys())
            group_mapping = {k: i for i, k in enumerate(group_set)}
            self.phone_to_group_mapping = {}
            for k, group in self.grouped_phones.items():
                for p in group:
                    self.phone_to_group_mapping[p] = group_mapping[k]
                    fst.add_arc(
                        initial_state,
                        pywrapfst.Arc(phone_mapping[p], group_mapping[k], 0, initial_state),
                    )
                processed.update(group)
            with mfa_open(group_mapping_path, "w") as f:
                for i, k in group_mapping.items():
                    f.write(f"{k} {i}\n")
            for phone, i in phone_mapping.items():
                if phone in processed:
                    continue
                fst.add_arc(initial_state, pywrapfst.Arc(i, i, 0, initial_state))
            fst.arcsort("olabel")
            fst.write(lexicon_path)
            min_length = round(self.frame_shift_seconds / 3, 4)

            for d_id, cmvn_path in self.cmvn_paths.items():
                wav_path = os.path.join(
                    self.working_directory, f"fine_tune_wav.{d_id}.{self.job_name}.scp"
                )
                segment_path = os.path.join(
                    self.working_directory, f"fine_tune_segments.{d_id}.{self.job_name}.scp"
                )

                utt2spk_path = os.path.join(
                    self.working_directory, f"fine_tune_utt2spk.{d_id}.{self.job_name}.scp"
                )
                text_path = os.path.join(
                    self.working_directory, f"fine_tune_text.{d_id}.{self.job_name}.scp"
                )
                fst_ark_path = os.path.join(
                    self.working_directory, f"fine_tune_fsts.{d_id}.{self.job_name}.ark"
                )
                feature_segment_path = os.path.join(
                    self.working_directory,
                    f"fine_tune_feature_segments.{d_id}.{self.job_name}.scp",
                )

                fmllr_path = self.fmllr_paths[d_id]
                self.setup_files(session, d_id, phone_mapping)

                proc = subprocess.Popen(
                    [
                        thirdparty_binary("compile-train-graphs"),
                        f"--read-disambig-syms={self.disambiguation_symbols_int_path}",
                        self.tree_path,
                        self.model_path,
                        lexicon_path,
                        f"ark:{text_path}",
                        f"ark:{fst_ark_path}",
                    ],
                    stderr=log_file,
                    env=os.environ,
                )
                proc.communicate()

                paste_proc, comp_proc = compute_feature_process(
                    log_file,
                    wav_path,
                    segment_path,
                    self.mfcc_options,
                    self.pitch_options,
                    min_length=min_length,
                    no_logging=True,
                )
                feature_proc = paste_proc if paste_proc is not None else comp_proc
                extract_proc = subprocess.Popen(
                    [
                        thirdparty_binary("extract-feature-segments"),
                        f"--min-segment-length={min_length}",
                        f"--frame-shift={self.new_frame_shift}",
                        f'--snip-edges={self.mfcc_options["snip-edges"]}',
                        "ark:-",
                        feature_segment_path,
                        "ark:-",
                    ],
                    stdin=feature_proc.stdout,
                    stderr=log_file,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                )
                trans_proc = compute_transform_process(
                    log_file,
                    extract_proc,
                    utt2spk_path,
                    cmvn_path,
                    self.lda_mat_path,
                    fmllr_path,
                    self.lda_options,
                )
                align_proc = subprocess.Popen(
                    [
                        thirdparty_binary("gmm-align-compiled"),
                        f"--transition-scale={self.align_options['transition_scale']}",
                        f"--acoustic-scale={self.align_options['acoustic_scale']}",
                        f"--self-loop-scale={self.align_options['self_loop_scale']}",
                        f"--beam={self.align_options['beam']}",
                        f"--retry-beam={self.align_options['retry_beam']}",
                        "--careful=false",
                        self.model_path,
                        f"ark,s,cs:{fst_ark_path}",
                        "ark,s,cs:-",
                        "ark:-",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    stdin=trans_proc.stdout,
                    env=os.environ,
                )

                ctm_proc = subprocess.Popen(
                    [
                        thirdparty_binary("ali-to-phones"),
                        "--ctm-output",
                        f"--frame-shift={self.new_frame_shift_seconds}",
                        self.model_path,
                        "ark,s,cs:-",
                        "-",
                    ],
                    stderr=log_file,
                    stdin=align_proc.stdout,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                    encoding="utf8",
                )
                interval_mapping = []
                current_utterance = None
                for boundary_id, ctm_intervals in parse_ctm_output(
                    ctm_proc, reversed_phone_mapping, raw_id=True
                ):
                    utterance_id, interval_id = boundary_id.split("-")
                    interval_id = int(interval_id)
                    utterance_id = int(utterance_id)

                    if current_utterance is None:
                        current_utterance = utterance_id
                    if current_utterance != utterance_id:
                        interval_mapping = sorted(interval_mapping, key=lambda x: x["id"])
                        interval_mapping.insert(
                            0, self.utterance_initial_intervals[current_utterance]
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
                        yield interval_mapping, deletions
                        interval_mapping = []
                        current_utterance = utterance_id
                    interval_mapping.append(
                        {
                            "id": interval_id,
                            "begin": round(
                                ctm_intervals[1].begin + self.segment_begins[interval_id], 4
                            ),
                            "end": self.original_intervals[interval_id]["end"],
                            "label": ctm_intervals[1].label,
                        }
                    )
                if interval_mapping:
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
                    yield interval_mapping, deletions
                self.check_call(ctm_proc)


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
        self.model_path = args.model_path
        self.phone_pdf_counts_path = args.phone_pdf_counts_path
        self.feature_strings = args.feature_strings

    def _run(self) -> typing.Generator[typing.Tuple[int, str]]:
        """Run the function"""
        db_engine = sqlalchemy.create_engine(self.db_string)
        with Session(db_engine) as session:
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
            phone_mapping = {p.phone: p.id for p in session.query(Phone)}

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

        with mfa_open(self.log_path, "w") as log_file:
            for dict_id in self.feature_strings.keys():
                feature_string = self.feature_strings[dict_id]
                output_proc = subprocess.Popen(
                    [
                        thirdparty_binary("gmm-compute-likes"),
                        self.model_path,
                        feature_string,
                        "ark,t:-",
                    ],
                    stderr=log_file,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                )
                interval_mappings = []
                new_interval_mappings = []
                for utterance_id, likelihoods in read_feats(output_proc):
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
                        alternate_label = max(alternate_labels, key=lambda x: alternate_labels[x])
                        interval_mappings.append({"id": pi.id, "phone_goodness": average_score})
                        new_interval_mappings.append(
                            {
                                "begin": pi.begin,
                                "end": pi.end,
                                "utterance_id": pi.utterance_id,
                                "phone_id": phone_mapping[alternate_label],
                            }
                        )
                    yield interval_mappings
                    interval_mappings = []
                self.check_call(output_proc)


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
        self.text_int_paths = args.text_int_paths
        self.ali_paths = args.ali_paths
        self.model_path = args.model_path
        self.for_g2p = args.for_g2p
        self.reversed_phone_mapping = {}
        self.phone_mapping = {}
        self.reversed_word_mapping = {}
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
        db_engine = sqlalchemy.create_engine(self.db_string)
        align_lexicon_paths = {}
        with mfa_open(self.log_path, "w") as log_file, Session(db_engine) as session:
            phones = session.query(Phone.phone, Phone.mapping_id)
            for phone, mapping_id in phones:
                self.reversed_phone_mapping[mapping_id] = phone
                self.phone_mapping[phone] = mapping_id
            for dict_id in self.text_int_paths.keys():
                d = session.query(Dictionary).get(dict_id)
                self.clitic_marker = d.clitic_marker
                self.silence_words.add(d.silence_word)
                self.oov_word = d.oov_word
                self.optional_silence_phone = d.optional_silence_phone
                align_lexicon_paths[dict_id] = d.align_lexicon_path
                self.reversed_word_mapping[d.id] = {}

                silence_words = (
                    session.query(Word.word)
                    .filter(Word.dictionary_id == dict_id)
                    .filter(Word.word_type == WordType.silence)
                )
                self.silence_words.update(x for x, in silence_words)

                words = session.query(Word.mapping_id, Word.word).filter(
                    Word.dictionary_id == dict_id
                )
                for w_id, w in words:
                    self.reversed_word_mapping[d.id][w_id] = w
                text_int_path = self.text_int_paths[dict_id]
                ali_path = self.ali_paths[dict_id]
                if not os.path.exists(ali_path):
                    continue

                phones_proc = subprocess.Popen(
                    [
                        thirdparty_binary("ali-to-phones"),
                        self.model_path,
                        f"ark:{ali_path}",
                        "ark,t:-",
                    ],
                    stderr=log_file,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                    encoding="utf8",
                )
                prons_proc = subprocess.Popen(
                    [
                        thirdparty_binary("phones-to-prons"),
                        align_lexicon_paths[dict_id],
                        str(self.phone_mapping["#1"]),
                        str(self.phone_mapping["#2"]),
                        "ark:-",
                        f"ark:{text_int_path}",
                        "ark,t:-",
                    ],
                    stdin=phones_proc.stdout,
                    stderr=log_file,
                    encoding="utf8",
                    stdout=subprocess.PIPE,
                    env=os.environ,
                )
                for line in prons_proc.stdout:
                    utt, prons_line = line.strip().split(maxsplit=1)
                    prons = prons_line.split(";")
                    word_pronunciations = []
                    for pron in prons:
                        pron = pron.strip()
                        if not pron:
                            continue
                        pron = pron.split()
                        word = pron.pop(0)
                        word = self.reversed_word_mapping[dict_id][int(word)]
                        pron = [self.reversed_phone_mapping[int(x)] for x in pron]
                        word_pronunciations.append((word, " ".join(pron)))
                    if self.for_g2p:
                        phones = []
                        for x in word_pronunciations:
                            phones.append("#1")
                            phones.extend(x[1].split())
                            phones.append("#2")
                        yield dict_id, utt, " ".join(phones)
                    else:
                        yield dict_id, self._process_pronunciations(word_pronunciations)
                self.check_call(prons_proc)


def compile_information_func(
    arguments: CompileInformationArguments,
) -> Dict[str, Union[List[str], float, int]]:
    """
    Multiprocessing function for compiling information about alignment

    See Also
    --------
    :meth:`.AlignMixin.compile_information`
        Main function that calls this function in parallel

    Parameters
    ----------
    arguments: CompileInformationArguments
        Arguments for the function

    Returns
    -------
    dict[str, Union[list[str], float, int]]
        Information about log-likelihood and number of unaligned files
    """
    average_logdet_pattern = re.compile(
        r"Overall average logdet is (?P<logdet>[-.,\d]+) over (?P<frames>[.\d+e]+) frames"
    )
    log_like_pattern = re.compile(
        r"^LOG .* Overall log-likelihood per frame is (?P<log_like>[-0-9.]+) over (?P<frames>\d+) frames.*$"
    )

    decode_error_pattern = re.compile(
        r"^WARNING .* Did not successfully decode file (?P<utt>.*?), .*$"
    )

    data = {"unaligned": [], "too_short": [], "log_like": 0, "total_frames": 0}
    align_log_path = arguments.align_log_path
    if not os.path.exists(align_log_path):
        align_log_path = align_log_path.replace(".log", ".fmllr.log")
    with mfa_open(arguments.log_path, "w"), mfa_open(align_log_path, "r") as f:
        for line in f:
            decode_error_match = re.match(decode_error_pattern, line)
            if decode_error_match:
                data["unaligned"].append(decode_error_match.group("utt"))
                continue
            log_like_match = re.match(log_like_pattern, line)
            if log_like_match:
                log_like = log_like_match.group("log_like")
                frames = log_like_match.group("frames")
                data["log_like"] = float(log_like)
                data["total_frames"] = int(frames)
            m = re.search(average_logdet_pattern, line)
            if m:
                logdet = float(m.group("logdet"))
                frames = float(m.group("frames"))
                data["logdet"] = logdet
                data["logdet_frames"] = frames
    return data


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
        self.model_path = args.model_path
        self.frame_shift = args.frame_shift
        self.ali_paths = args.ali_paths
        self.text_int_paths = args.text_int_paths
        self.utterance_begins = {}
        self.reversed_phone_mapping = {}
        self.reversed_word_mapping = {}
        self.pronunciation_mapping = {}
        self.phone_mapping = {}
        self.silence_words = set()

    def cleanup_intervals(
        self,
        utterance_name,
        intervals: List[CtmInterval],
        word_pronunciations: List[typing.Tuple[str, List[str]]],
    ):
        """
        Clean up phone intervals to remove silence

        Parameters
        ----------
        intervals: list[:class:`~montreal_forced_aligner.data.CtmInterval`]
            Intervals to process

        Returns
        -------
        list[:class:`~montreal_forced_aligner.data.CtmInterval`]
            Cleaned up intervals
        """
        actual_phone_intervals = []
        actual_word_intervals = []
        phone_word_mapping = []
        utterance_begin = self.utterance_begins[utterance_name]
        current_word_begin = None
        words_index = 0
        current_phones = []
        for interval in intervals:
            interval.begin += utterance_begin
            interval.end += utterance_begin
            if interval.label == self.optional_silence_phone:
                interval.label = self.phone_to_phone_id[interval.label]
                cur_word = word_pronunciations[words_index]
                actual_phone_intervals.append(interval)
                actual_word_intervals.append(
                    WordCtmInterval(
                        interval.begin,
                        interval.end,
                        word_pronunciations[words_index][0],
                        self.pronunciation_mapping[(cur_word[0], " ".join(cur_word[1]))],
                    )
                )
                phone_word_mapping.append(len(actual_word_intervals) - 1)
                current_word_begin = None
                current_phones = []
                words_index += 1
                continue
            if current_word_begin is None:
                current_word_begin = interval.begin
            current_phones.append(interval.label)
            cur_word = word_pronunciations[words_index]
            if current_phones == cur_word[1]:
                actual_word_intervals.append(
                    WordCtmInterval(
                        current_word_begin,
                        interval.end,
                        cur_word[0],
                        self.pronunciation_mapping[(cur_word[0], " ".join(cur_word[1]))],
                    )
                )
                for _ in range(len(current_phones)):
                    phone_word_mapping.append(len(actual_word_intervals) - 1)
                current_word_begin = None
                current_phones = []
                words_index += 1
            interval.label = self.phone_to_phone_id[interval.label]
            actual_phone_intervals.append(interval)
        return actual_word_intervals, actual_phone_intervals, phone_word_mapping

    def _run(self) -> typing.Generator[typing.Tuple[int, List[CtmInterval], List[CtmInterval]]]:
        """Run the function"""
        db_engine = sqlalchemy.create_engine(self.db_string)
        align_lexicon_paths = {}
        with Session(db_engine) as session:
            for dict_id in self.ali_paths.keys():
                d = session.query(Dictionary).get(dict_id)

                self.clitic_marker = d.clitic_marker
                self.silence_word = d.silence_word
                self.oov_word = d.oov_word
                self.optional_silence_phone = d.optional_silence_phone
                align_lexicon_paths[dict_id] = d.align_lexicon_path
                silence_words = (
                    session.query(Word.id)
                    .filter(Word.dictionary_id == dict_id)
                    .filter(Word.word_type == WordType.silence)
                )
                self.silence_words.update(x for x, in silence_words)

                words = session.query(Word.mapping_id, Word.id).filter(
                    Word.dictionary_id == dict_id
                )
                self.reversed_word_mapping[dict_id] = {}
                for m_id, w_id in words:
                    self.reversed_word_mapping[dict_id][m_id] = w_id
            utts = session.query(Utterance.id, Utterance.begin).filter(
                Utterance.job_id == self.job_name
            )
            for u_id, begin in utts:
                self.utterance_begins[u_id] = begin
            self.phone_to_phone_id = {}
            ds = session.query(Phone.phone, Phone.id, Phone.mapping_id).all()
            for phone, p_id, mapping_id in ds:
                self.reversed_phone_mapping[mapping_id] = phone
                self.phone_to_phone_id[phone] = p_id
                self.phone_mapping[phone] = mapping_id

            pronunciations = (
                session.query(Word.id, Pronunciation.pronunciation, Pronunciation.id)
                .join(Pronunciation.word)
                .filter(Word.dictionary_id.in_(self.ali_paths.keys()))
            )
            for w_id, pron, p_id in pronunciations:
                self.pronunciation_mapping[(w_id, pron)] = p_id
        with mfa_open(self.log_path, "w") as log_file:
            for dict_id in self.ali_paths.keys():
                ali_path = self.ali_paths[dict_id]
                text_int_path = self.text_int_paths[dict_id]

                ctm_proc = subprocess.Popen(
                    [
                        thirdparty_binary("ali-to-phones"),
                        "--ctm-output",
                        f"--frame-shift={self.frame_shift}",
                        self.model_path,
                        f"ark:{ali_path}",
                        "-",
                    ],
                    stderr=log_file,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                    encoding="utf8",
                )

                phones_proc = subprocess.Popen(
                    [
                        thirdparty_binary("ali-to-phones"),
                        self.model_path,
                        f"ark:{ali_path}",
                        "ark,t:-",
                    ],
                    stderr=log_file,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                    encoding="utf8",
                )
                prons_proc = subprocess.Popen(
                    [
                        thirdparty_binary("phones-to-prons"),
                        align_lexicon_paths[dict_id],
                        str(self.phone_mapping["#1"]),
                        str(self.phone_mapping["#2"]),
                        "ark:-",
                        f"ark:{text_int_path}",
                        "ark,t:-",
                    ],
                    stdin=phones_proc.stdout,
                    stderr=log_file,
                    encoding="utf8",
                    stdout=subprocess.PIPE,
                    env=os.environ,
                )
                for utterance, intervals in parse_ctm_output(
                    ctm_proc, self.reversed_phone_mapping
                ):
                    while True:
                        prons_line = prons_proc.stdout.readline().strip()
                        if prons_line:
                            break
                    utt_id, prons_line = prons_line.split(maxsplit=1)
                    prons = prons_line.split(";")
                    word_pronunciations = []
                    for pron in prons:
                        pron = pron.strip()
                        if not pron:
                            continue
                        pron = pron.split()
                        word = pron.pop(0)
                        word = self.reversed_word_mapping[dict_id][int(word)]
                        pron = [self.reversed_phone_mapping[int(x)] for x in pron]
                        word_pronunciations.append((word, pron))
                    word_intervals, phone_intervals, phone_word_mapping = self.cleanup_intervals(
                        utterance, intervals, word_pronunciations
                    )
                    yield utterance, word_intervals, phone_intervals, phone_word_mapping
                self.check_call(ctm_proc)


def construct_output_tiers(
    session: Session,
    file_id: int,
    workflow_id: int,
    cleanup_textgrids: bool,
    clitic_marker: str,
    include_original_text: bool,
) -> Dict[str, Dict[str, List[CtmInterval]]]:
    """
    Construct aligned output tiers for a file

    Parameters
    ----------
    session: Session
        SqlAlchemy session
    file_id: int
        Integer ID for the file

    Returns
    -------
    Dict[str, Dict[str,List[CtmInterval]]]
        Aligned tiers
    """
    utterances = (
        session.query(Utterance)
        .options(
            joinedload(Utterance.speaker, innerjoin=True).load_only(Speaker.name),
        )
        .filter(Utterance.file_id == file_id)
    )
    workflow = session.query(CorpusWorkflow).get(workflow_id)
    data = {}
    for utt in utterances:
        word_intervals = (
            session.query(WordInterval, Word)
            .join(WordInterval.word)
            .filter(WordInterval.utterance_id == utt.id)
            .filter(WordInterval.workflow_id == workflow_id)
            .options(
                selectinload(WordInterval.phone_intervals).joinedload(
                    PhoneInterval.phone, innerjoin=True
                )
            )
            .order_by(WordInterval.begin)
        )
        if cleanup_textgrids:
            word_intervals = word_intervals.filter(Word.word_type != WordType.silence)
        if utt.speaker.name not in data:
            data[utt.speaker.name] = {"words": [], "phones": []}
            if include_original_text:
                data[utt.speaker.name]["utterances"] = []
        actual_words = utt.normalized_text.split()
        if include_original_text:
            data[utt.speaker.name]["utterances"].append(CtmInterval(utt.begin, utt.end, utt.text))
        for i, (wi, w) in enumerate(word_intervals.all()):
            if len(wi.phone_intervals) == 0:
                continue
            label = w.word
            if cleanup_textgrids:
                if w.word_type is WordType.oov and workflow.workflow is WorkflowType.alignment:
                    label = actual_words[i]
                if (
                    data[utt.speaker.name]["words"]
                    and clitic_marker
                    and data[utt.speaker.name]["words"][-1].label.endswith(clitic_marker)
                    or label.startswith(clitic_marker)
                ):
                    data[utt.speaker.name]["words"][-1].end = wi.end
                    data[utt.speaker.name]["words"][-1].label += label

                    for pi in wi.phone_intervals:
                        data[utt.speaker.name]["phones"].append(
                            CtmInterval(pi.begin, pi.end, pi.phone.phone)
                        )
                    continue

            data[utt.speaker.name]["words"].append(CtmInterval(wi.begin, wi.end, label))

            for pi in wi.phone_intervals:
                data[utt.speaker.name]["phones"].append(
                    CtmInterval(pi.begin, pi.end, pi.phone.phone)
                )
    return data


def construct_output_path(
    name: str,
    relative_path: str,
    output_directory: str,
    input_path: str = "",
    output_format: str = TextgridFormats.SHORT_TEXTGRID,
) -> str:
    """
    Construct an output path

    Returns
    -------
    str
        Output path
    """
    if output_format.upper() == "LAB":
        extension = ".lab"
    elif output_format.upper() == "JSON":
        extension = ".json"
    elif output_format.upper() == "CSV":
        extension = ".csv"
    else:
        extension = ".TextGrid"
    if relative_path:
        print(output_directory, relative_path)
        relative = os.path.join(output_directory, relative_path)
    else:
        relative = output_directory
    output_path = os.path.join(relative, name + extension)
    if output_path == input_path:
        output_path = os.path.join(relative, name + "_aligned" + extension)
    os.makedirs(relative, exist_ok=True)
    return output_path


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
    stopped: :class:`~montreal_forced_aligner.utils.Stopped`
        Stop check for processing
    finished_processing: :class:`~montreal_forced_aligner.utils.Stopped`
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
        stopped: Stopped,
        finished_adding: Stopped,
        arguments: ExportTextGridArguments,
        exported_file_count: Counter,
    ):
        mp.Process.__init__(self)
        self.db_string = db_string
        self.for_write_queue = for_write_queue
        self.return_queue = return_queue
        self.stopped = stopped
        self.finished_adding = finished_adding
        self.finished_processing = Stopped()

        self.output_directory = arguments.output_directory
        self.output_format = arguments.output_format
        self.export_frame_shift = arguments.export_frame_shift
        self.log_path = arguments.log_path
        self.include_original_text = arguments.include_original_text
        self.workflow_id = arguments.workflow_id
        self.cleanup_textgrids = arguments.cleanup_textgrids
        self.clitic_marker = arguments.clitic_marker
        self.exported_file_count = exported_file_count

    def run(self) -> None:
        """Run the exporter function"""
        db_engine = sqlalchemy.create_engine(self.db_string)
        with mfa_open(self.log_path, "w") as log_file, Session(db_engine) as session:
            log_file.write(f"Exporting TextGrids for Workflow ID: {self.workflow_id}\n")
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
                    if self.finished_adding.stop_check():
                        self.finished_processing.stop()
                        break
                    continue

                if self.stopped.stop_check():
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
                        self.workflow_id,
                        self.cleanup_textgrids,
                        self.clitic_marker,
                        self.include_original_text,
                    )
                    export_textgrid(
                        data, output_path, duration, self.export_frame_shift, self.output_format
                    )
                    self.return_queue.put(1)
                except Exception:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    self.return_queue.put(
                        AlignmentExportError(
                            output_path,
                            traceback.format_exception(exc_type, exc_value, exc_traceback),
                        )
                    )
                    self.stopped.stop()
                    raise
            log_file.write("Done!\n")


class TranscriptionAlignmentExtractionFunction(KaldiFunction):
    """
    Multiprocessing function for scoring lattices

    See Also
    --------
    :meth:`.AlignmentExtractionArguments._collect_alignments`
        Main function that calls this function in parallel
    :meth:`.CorpusAligner.AlignmentExtractionArguments`
        Job method for generating arguments for this function
    :kaldi_src:`lattice-scale`
        Relevant Kaldi binary
    :kaldi_src:`lattice-add-penalty`
        Relevant Kaldi binary
    :kaldi_src:`lattice-best-path`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.alignment.multiprocessing.AlignmentExtractionArguments`
        Arguments for the function
    """

    def __init__(self, args: AlignmentExtractionArguments):
        super().__init__(args)
        self.score_options = args.score_options
        self.lat_paths = args.ali_paths
        self.phone_symbol_path = args.phone_symbol_path

        self.model_path = args.model_path
        self.frame_shift = args.frame_shift
        self.utterance_begins = {}

    def _run(self) -> typing.Generator[typing.Tuple[str, float, float, float, int]]:
        """Run the function"""
        db_engine = sqlalchemy.create_engine(self.db_string)
        with Session(db_engine) as session, mfa_open(self.log_path, "w") as log_file:
            phones = session.query(Phone.id, Phone.mapping_id)
            reversed_phone_mapping = {}
            for p_id, m_id in phones:
                reversed_phone_mapping[m_id] = p_id
            utts = (
                session.query(Utterance)
                .join(Utterance.speaker)
                .filter(Utterance.job_id == self.job_name)
                .options(load_only(Utterance.id, Utterance.begin))
            )
            for utt in utts:
                self.utterance_begins[utt.id] = utt.begin
            for dict_id in self.lat_paths.keys():
                language_model_weight = self.score_options["language_model_weight"]
                lat_path = self.lat_paths[dict_id]

                one_best_proc = subprocess.Popen(
                    [
                        thirdparty_binary("lattice-1best"),
                        f"--acoustic-scale={language_model_weight/100}",
                        f"ark:{lat_path}",
                        "ark:-",
                    ],
                    stderr=log_file,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                )

                linear_proc = subprocess.Popen(
                    [
                        thirdparty_binary("nbest-to-linear"),
                        "ark:-",
                        "ark:-",
                    ],
                    stderr=log_file,
                    stdin=one_best_proc.stdout,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                )

                ctm_proc = subprocess.Popen(
                    [
                        thirdparty_binary("ali-to-phones"),
                        "--ctm-output",
                        self.model_path,
                        "ark:-",
                        "-",
                    ],
                    stderr=log_file,
                    stdin=linear_proc.stdout,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                    encoding="utf8",
                )
                for utterance, intervals in parse_ctm_output(ctm_proc, reversed_phone_mapping):
                    for i in intervals:
                        i.begin += self.utterance_begins[utterance]
                        i.end += self.utterance_begins[utterance]
                    yield utterance, [], intervals, []
                self.check_call(ctm_proc)
