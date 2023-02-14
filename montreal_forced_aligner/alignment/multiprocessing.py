"""
Alignment multiprocessing functions
-----------------------------------

"""
from __future__ import annotations

import collections
import json
import logging
import multiprocessing as mp
import os
import re
import statistics
import subprocess
import sys
import traceback
import typing
from pathlib import Path
from queue import Empty
from typing import TYPE_CHECKING, Dict, List, Union

import numpy as np
import pynini
import pywrapfst
import sqlalchemy
from sqlalchemy.orm import Session, joinedload, selectinload, subqueryload

from montreal_forced_aligner.corpus.features import (
    compute_mfcc_process,
    compute_pitch_process,
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
    File,
    Job,
    Phone,
    PhoneInterval,
    Pronunciation,
    SoundFile,
    Speaker,
    Utterance,
    Word,
    WordInterval,
)
from montreal_forced_aligner.exceptions import AlignmentExportError, FeatureGenerationError
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


def phones_to_prons(
    text: str,
    intervals: List[CtmInterval],
    align_lexicon_fst: pynini.Fst,
    word_symbol_table: pywrapfst.SymbolTableView,
    phone_symbol_table: pywrapfst.SymbolTableView,
    optional_silence_phone: str,
    transcription: bool = False,
    clitic_marker=None,
):
    if "<space>" in text:
        words = [x.replace(" ", "") for x in text.split("<space>")]
    else:
        words = text.split()
    word_begin = "#1"
    word_end = "#2"
    word_begin_symbol = phone_symbol_table.find(word_begin)
    word_end_symbol = phone_symbol_table.find(word_end)
    acceptor = pynini.accep(text, token_type=word_symbol_table)
    phone_to_word = pynini.compose(align_lexicon_fst, acceptor)
    phone_fst = pynini.Fst()
    current_state = phone_fst.add_state()
    phone_fst.set_start(current_state)
    for p in intervals:
        next_state = phone_fst.add_state()
        symbol = phone_symbol_table.find(p.label)
        phone_fst.add_arc(
            current_state,
            pywrapfst.Arc(
                symbol, symbol, pywrapfst.Weight.one(phone_fst.weight_type()), next_state
            ),
        )
        current_state = next_state
    if transcription:
        if intervals[-1].label == optional_silence_phone:
            state = current_state - 1
        else:
            state = current_state
        phone_to_word_state = phone_to_word.num_states() - 1
        for i in range(phone_symbol_table.num_symbols()):
            if phone_symbol_table.find(i) == "<eps>":
                continue
            if phone_symbol_table.find(i).startswith("#"):
                continue
            phone_fst.add_arc(
                state,
                pywrapfst.Arc(
                    phone_symbol_table.find("<eps>"),
                    i,
                    pywrapfst.Weight.one(phone_fst.weight_type()),
                    state,
                ),
            )

            phone_to_word.add_arc(
                phone_to_word_state,
                pywrapfst.Arc(
                    i,
                    phone_symbol_table.find("<eps>"),
                    pywrapfst.Weight.one(phone_fst.weight_type()),
                    phone_to_word_state,
                ),
            )
    for s in range(current_state + 1):
        phone_fst.add_arc(
            s,
            pywrapfst.Arc(
                word_end_symbol, word_end_symbol, pywrapfst.Weight.one(phone_fst.weight_type()), s
            ),
        )
        phone_fst.add_arc(
            s,
            pywrapfst.Arc(
                word_begin_symbol,
                word_begin_symbol,
                pywrapfst.Weight.one(phone_fst.weight_type()),
                s,
            ),
        )

    phone_fst.set_final(current_state, pywrapfst.Weight.one(phone_fst.weight_type()))
    phone_fst.arcsort("olabel")

    lattice = pynini.compose(phone_fst, phone_to_word)
    try:
        path_string = pynini.shortestpath(lattice).project("input").string(phone_symbol_table)
    except Exception:
        logging.debug("For the text and intervals:")
        logging.debug(text)
        logging.debug([x.label for x in intervals])
        logging.debug("There was an issue composing word and phone FSTs")
        logging.debug("PHONE FST:")
        phone_fst.set_input_symbols(phone_symbol_table)
        phone_fst.set_output_symbols(phone_symbol_table)
        logging.debug(phone_fst)
        logging.debug("PHONE_TO_WORD FST:")
        phone_to_word.set_input_symbols(phone_symbol_table)
        phone_to_word.set_output_symbols(word_symbol_table)
        logging.debug(phone_to_word)
        raise
    path_string = path_string.replace(f"{word_end} {word_begin}", word_begin)
    path_string = path_string.replace(f"{word_end}", word_begin)
    word_splits = re.split(rf" ?{word_begin} ?", path_string)
    word_splits = [x.split() for x in word_splits if x != optional_silence_phone and x]

    return list(zip(words, word_splits))


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
    phone_symbol_path: str
    score_options: MetaDict
    confidence: bool
    transcription: bool


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

    tree_path: str
    model_path: str
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

    model_path: str
    align_options: MetaDict
    feature_options: MetaDict
    confidence: bool


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

    phone_symbol_table_path: str
    disambiguation_symbols_int_path: str
    tree_path: str
    model_path: str
    frame_shift: int
    mfcc_options: MetaDict
    pitch_options: MetaDict
    lda_options: MetaDict
    align_options: MetaDict
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
        self.tree_path = args.tree_path
        self.model_path = args.model_path
        self.use_g2p = args.use_g2p

    def _run(self) -> typing.Generator[typing.Tuple[int, int]]:
        """Run the function"""

        with mfa_open(self.log_path, "w") as log_file, Session(self.db_engine()) as session:
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
            out_disambig = os.path.join(workflow.working_directory, f"{self.job_name}.disambig")
            ilabels_temp = os.path.join(workflow.working_directory, f"{self.job_name}.ilabels")
            clg_path = os.path.join(workflow.working_directory, f"{self.job_name}.clg.temp")
            ha_out_disambig = os.path.join(
                workflow.working_directory, f"{self.job_name}.ha_out_disambig.temp"
            )
            text_int_paths = job.per_dictionary_text_int_scp_paths
            if self.use_g2p:
                import pynini
                from pynini.lib import rewrite

                from montreal_forced_aligner.g2p.generator import threshold_lattice_to_dfa

                for d in job.dictionaries:
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
                    fst_ark_path = job.construct_path(
                        workflow.working_directory, "fsts", "ark", d.id
                    )

                    with mfa_open(fst_ark_path, "wb") as fst_output_file:
                        for utt_id, full_text in utterances:
                            try:
                                lattice = rewrite.rewrite_lattice(full_text, fst, token_type)
                                lattice = threshold_lattice_to_dfa(lattice, 2.0)
                                input = lattice.write_to_string()
                            except pynini.lib.rewrite.Error:
                                log_file.write(f'Error composing "{full_text}"\n')
                                log_file.flush()
                                continue
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
                for d in job.dictionaries:
                    fst_ark_path = job.construct_path(
                        workflow.working_directory, "fsts", "ark", d.id
                    )
                    text_path = text_int_paths[d.id]
                    proc = subprocess.Popen(
                        [
                            thirdparty_binary("compile-train-graphs"),
                            f"--read-disambig-syms={d.disambiguation_symbols_int_path}",
                            self.tree_path,
                            self.model_path,
                            d.lexicon_fst_path,
                            f"ark,s,cs:{text_path}",
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
    :kaldi_src:`align-gmm-compiled`
        Relevant Kaldi binary
    :kaldi_src:`gmm-boost-silence`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.alignment.multiprocessing.AlignArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(
        r"^LOG.*Log-like per frame for utterance (?P<utterance>.*) is (?P<loglike>[-\d.]+) over (?P<num_frames>\d+) frames."
    )

    def __init__(self, args: AlignArguments):
        super().__init__(args)
        self.model_path = args.model_path
        self.align_options = args.align_options
        self.feature_options = args.feature_options
        self.confidence = args.confidence

    def _run(self) -> typing.Generator[typing.Tuple[int, float]]:
        """Run the function"""

        with mfa_open(self.log_path, "w") as log_file, Session(self.db_engine()) as session:
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
                dict_id = d.id
                word_symbols_path = d.words_symbol_path
                feature_string = job.construct_feature_proc_string(
                    workflow.working_directory,
                    dict_id,
                    self.feature_options["uses_splices"],
                    self.feature_options["splice_left_context"],
                    self.feature_options["splice_right_context"],
                    self.feature_options["uses_speaker_adaptation"],
                )
                fst_path = job.construct_path(workflow.working_directory, "fsts", "ark", dict_id)
                fmllr_path = job.construct_path(
                    workflow.working_directory, "trans", "ark", dict_id
                )
                ali_path = job.construct_path(workflow.working_directory, "ali", "ark", dict_id)
                if (
                    self.confidence
                    and self.feature_options["uses_speaker_adaptation"]
                    and os.path.exists(fmllr_path)
                ):
                    ali_path = job.construct_path(
                        workflow.working_directory, "lat", "ark", dict_id
                    )
                    com = [
                        thirdparty_binary("gmm-latgen-faster"),
                        f"--acoustic-scale={self.align_options['acoustic_scale']}",
                        f"--beam={self.align_options['beam']}",
                        f"--max-active={self.align_options['max_active']}",
                        f"--lattice-beam={self.align_options['lattice_beam']}",
                        f"--word-symbol-table={word_symbols_path}",
                        self.model_path,
                        f"ark,s,cs:{fst_path}",
                        feature_string,
                        f"ark:{ali_path}",
                    ]
                    align_proc = subprocess.Popen(
                        com, stderr=subprocess.PIPE, env=os.environ, encoding="utf8"
                    )
                    process_stream = align_proc.stderr
                else:
                    com = [
                        thirdparty_binary("gmm-align-compiled"),
                        f"--transition-scale={self.align_options['transition_scale']}",
                        f"--acoustic-scale={self.align_options['acoustic_scale']}",
                        f"--self-loop-scale={self.align_options['self_loop_scale']}",
                        f"--beam={self.align_options['beam']}",
                        f"--retry-beam={self.align_options['retry_beam']}",
                        "--careful=false",
                        "-",
                        f"ark,s,cs:{fst_path}",
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
                    process_stream = align_proc.stdout
                no_feature_count = 0
                for line in process_stream:
                    if re.search("No features for utterance", line):
                        no_feature_count += 1
                    line = line.strip()
                    if (
                        self.confidence
                        and self.feature_options["uses_speaker_adaptation"]
                        and os.path.exists(fmllr_path)
                    ):
                        m = self.progress_pattern.match(line)
                        if m:
                            utterance = m.group("utterance")
                            u_id = int(utterance.split("-")[-1])
                            yield u_id, float(m.group("loglike"))
                    else:
                        utterance, log_likelihood = line.split()
                        u_id = int(utterance.split("-")[-1])
                        yield u_id, float(log_likelihood)
                if no_feature_count:
                    align_proc.wait()
                    raise FeatureGenerationError(
                        f"There was an issue in feature generation for {no_feature_count} utterances. "
                        f"This can be caused by version incompatibilities between MFA and the model, "
                        f"in which case you should re-download or re-train your model, "
                        f"or downgrade MFA to the version that the model was trained on."
                    )
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
        self.tree_path = args.tree_path
        self.model_path = args.model_path
        self.mfcc_options = args.mfcc_options
        self.mfcc_options["frame-shift"] = self.new_frame_shift
        self.mfcc_options["snip-edges"] = False
        self.pitch_options = args.pitch_options
        self.pitch_options["frame-shift"] = self.new_frame_shift
        self.pitch_options["snip-edges"] = False
        self.lda_options = args.lda_options
        self.align_options = args.align_options
        self.grouped_phones = args.grouped_phones
        self.position_dependent_phones = args.position_dependent_phones
        self.disambiguation_symbols_int_path = args.disambiguation_symbols_int_path
        self.segment_begins = {}
        self.segment_ends = {}
        self.original_intervals = {}
        self.utterance_initial_intervals = {}

    def setup_files(
        self, session: Session, job: Job, workflow: CorpusWorkflow, dictionary_id: int
    ):
        wav_path = job.construct_path(
            workflow.working_directory, "fine_tune_wav", "scp", dictionary_id
        )
        segment_path = job.construct_path(
            workflow.working_directory, "fine_tune_segments", "scp", dictionary_id
        )
        feature_segment_path = job.construct_path(
            workflow.working_directory, "fine_tune_feature_segments", "scp", dictionary_id
        )
        utt2spk_path = job.construct_path(
            workflow.working_directory, "fine_tune_utt2spk", "scp", dictionary_id
        )
        text_path = job.construct_path(
            workflow.working_directory, "fine_tune_text", "scp", dictionary_id
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
            .filter(PhoneInterval.workflow_id == workflow.id)
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
        with Session(self.db_engine()) as session, mfa_open(self.log_path, "w") as log_file:
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

            reversed_phone_mapping = {}
            phone_mapping = {}
            phone_query = session.query(Phone.mapping_id, Phone.id, Phone.kaldi_label)
            for m_id, p_id, phone in phone_query:
                reversed_phone_mapping[m_id] = p_id
                phone_mapping[phone] = m_id

            lexicon_path = os.path.join(workflow.working_directory, "phone.fst")
            group_mapping_path = os.path.join(workflow.working_directory, "groups.txt")
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
            cmvn_paths = job.per_dictionary_cmvn_scp_paths
            for d_id in job.dictionary_ids:
                cmvn_path = cmvn_paths[d_id]
                wav_path = job.construct_path(
                    workflow.working_directory, "fine_tune_wav", "scp", d_id
                )
                segment_path = job.construct_path(
                    workflow.working_directory, "fine_tune_segments", "scp", d_id
                )
                feature_segment_path = job.construct_path(
                    workflow.working_directory, "fine_tune_feature_segments", "scp", d_id
                )
                utt2spk_path = job.construct_path(
                    workflow.working_directory, "fine_tune_utt2spk", "scp", d_id
                )
                text_path = job.construct_path(
                    workflow.working_directory, "fine_tune_text", "scp", d_id
                )
                pitch_ark_path = job.construct_path(
                    workflow.working_directory, "fine_tune_pitch", "ark", d_id
                )
                mfcc_ark_path = job.construct_path(
                    workflow.working_directory, "fine_tune_mfcc", "ark", d_id
                )
                feats_ark_path = job.construct_path(
                    workflow.working_directory, "fine_tune_feats", "ark", d_id
                )

                fmllr_path = job.construct_path(workflow.working_directory, "trans", "ark", d_id)

                self.setup_files(session, job, workflow, d_id)
                fst_ark_path = job.construct_path(
                    workflow.working_directory, "fine_tune_fsts", "ark", d_id
                )
                proc = subprocess.Popen(
                    [
                        thirdparty_binary("compile-train-graphs"),
                        f"--read-disambig-syms={self.disambiguation_symbols_int_path}",
                        self.tree_path,
                        self.model_path,
                        lexicon_path,
                        f"ark,s,cs:{text_path}",
                        f"ark:{fst_ark_path}",
                    ],
                    stderr=log_file,
                    env=os.environ,
                )
                proc.communicate()

                seg_proc = subprocess.Popen(
                    [
                        thirdparty_binary("extract-segments"),
                        f"--min-segment-length={min_length}",
                        f"scp:{wav_path}",
                        segment_path,
                        "ark:-",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=os.environ,
                )
                mfcc_proc = compute_mfcc_process(
                    log_file, wav_path, subprocess.PIPE, self.mfcc_options
                )
                cmvn_proc = subprocess.Popen(
                    [
                        "apply-cmvn",
                        f"--utt2spk=ark:{utt2spk_path}",
                        f"scp:{cmvn_path}",
                        "ark:-",
                        f"ark:{mfcc_ark_path}",
                    ],
                    env=os.environ,
                    stdin=mfcc_proc.stdout,
                    stderr=log_file,
                )

                use_pitch = self.pitch_options["use-pitch"] or self.pitch_options["use-voicing"]
                if use_pitch:
                    pitch_proc = compute_pitch_process(
                        log_file, wav_path, subprocess.PIPE, self.pitch_options
                    )
                    pitch_copy_proc = subprocess.Popen(
                        [
                            thirdparty_binary("copy-feats"),
                            "--compress=true",
                            "ark:-",
                            f"ark:{pitch_ark_path}",
                        ],
                        stdin=pitch_proc.stdout,
                        stderr=log_file,
                        env=os.environ,
                    )
                for line in seg_proc.stdout:
                    mfcc_proc.stdin.write(line)
                    mfcc_proc.stdin.flush()
                    if use_pitch:
                        pitch_proc.stdin.write(line)
                        pitch_proc.stdin.flush()
                mfcc_proc.stdin.close()
                if use_pitch:
                    pitch_proc.stdin.close()
                cmvn_proc.wait()
                if use_pitch:
                    pitch_copy_proc.wait()
                if use_pitch:
                    paste_proc = subprocess.Popen(
                        [
                            thirdparty_binary("paste-feats"),
                            "--length-tolerance=2",
                            f"ark:{mfcc_ark_path}",
                            f"ark:{pitch_ark_path}",
                            f"ark:{feats_ark_path}",
                        ],
                        stderr=log_file,
                        env=os.environ,
                    )
                    paste_proc.wait()
                else:
                    feats_ark_path = mfcc_ark_path

                extract_proc = subprocess.Popen(
                    [
                        thirdparty_binary("extract-feature-segments"),
                        f"--min-segment-length={min_length}",
                        f"--frame-shift={self.new_frame_shift}",
                        f'--snip-edges={self.mfcc_options["snip-edges"]}',
                        f"ark,s,cs:{feats_ark_path}",
                        feature_segment_path,
                        "ark:-",
                    ],
                    stdin=paste_proc.stdout,
                    stderr=log_file,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                )
                trans_proc = compute_transform_process(
                    log_file,
                    extract_proc,
                    utt2spk_path,
                    workflow.lda_mat_path,
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
        with Session(self.db_engine()) as session:
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
        self.model_path = args.model_path
        self.for_g2p = args.for_g2p
        self.reversed_phone_mapping = {}
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
        self.phone_symbol_table = None
        with mfa_open(self.log_path, "w") as log_file, Session(self.db_engine()) as session:
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
            phones = session.query(Phone.kaldi_label, Phone.mapping_id)
            for phone, mapping_id in phones:
                self.reversed_phone_mapping[mapping_id] = phone
            for d in job.dictionaries:
                utts = (
                    session.query(Utterance.id, Utterance.normalized_text)
                    .join(Utterance.speaker)
                    .filter(Utterance.job_id == self.job_name)
                    .filter(Speaker.dictionary_id == d.id)
                )
                self.utterance_texts = {}
                for u_id, text in utts:
                    self.utterance_texts[u_id] = text
                if self.phone_symbol_table is None:
                    self.phone_symbol_table = pywrapfst.SymbolTable.read_text(
                        d.phone_symbol_table_path
                    )
                self.word_symbol_table = pywrapfst.SymbolTable.read_text(d.words_symbol_path)
                self.align_lexicon_fst = pynini.Fst.read(d.align_lexicon_path)
                self.clitic_marker = d.clitic_marker
                self.silence_words.add(d.silence_word)
                self.oov_word = d.oov_word
                self.optional_silence_phone = d.optional_silence_phone
                self.oov_phone = d.oov_phone

                silence_words = (
                    session.query(Word.word)
                    .filter(Word.dictionary_id == d.id)
                    .filter(Word.word_type == WordType.silence)
                )
                self.silence_words.update(x for x, in silence_words)

                ali_path = job.construct_path(workflow.working_directory, "ali", "ark", d.id)
                if not os.path.exists(ali_path):
                    continue

                ctm_proc = subprocess.Popen(
                    [
                        thirdparty_binary("ali-to-phones"),
                        "--ctm-output",
                        self.model_path,
                        f"ark,s,cs:{ali_path}",
                        "-",
                    ],
                    stderr=log_file,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                    encoding="utf8",
                )
                for utterance, intervals in parse_ctm_output(
                    ctm_proc, self.reversed_phone_mapping
                ):
                    word_pronunciations = phones_to_prons(
                        self.utterance_texts[utterance],
                        intervals,
                        self.align_lexicon_fst,
                        self.word_symbol_table,
                        self.phone_symbol_table,
                        self.optional_silence_phone,
                    )
                    if d.position_dependent_phones:
                        word_pronunciations = [
                            (x[0], [split_phone_position(y)[0] for y in x[1]])
                            for x in word_pronunciations
                        ]
                    word_pronunciations = [(x[0], " ".join(x[1])) for x in word_pronunciations]
                    word_pronunciations = [
                        x if x[1] != self.oov_phone else (self.oov_word, self.oov_phone)
                        for x in word_pronunciations
                    ]
                    if self.for_g2p:
                        phones = []
                        for i, x in enumerate(word_pronunciations):
                            if i > 0 and (
                                x[0].startswith(self.clitic_marker)
                                or word_pronunciations[i - 1][0].endswith(self.clitic_marker)
                            ):
                                phones.pop(-1)
                            else:
                                phones.append("#1")
                            phones.extend(x[1].split())
                            phones.append("#2")
                        yield d.id, utterance, " ".join(phones)
                    else:
                        yield d.id, self._process_pronunciations(word_pronunciations)
                self.check_call(ctm_proc)


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
        self.utterance_begins = {}
        self.reversed_phone_mapping = {}
        self.reversed_word_mapping = {}
        self.pronunciation_mapping = {}
        self.phone_mapping = {}
        self.silence_words = set()
        self.confidence = args.confidence
        self.transcription = args.transcription
        self.score_options = args.score_options

    def cleanup_intervals(
        self,
        utterance_name,
        intervals: List[CtmInterval],
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
        word_pronunciations = phones_to_prons(
            self.utterance_texts[utterance_name],
            intervals,
            self.align_lexicon_fst,
            self.word_symbol_table,
            self.phone_symbol_table,
            self.optional_silence_phone,
            self.transcription,
        )
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
                actual_phone_intervals.append(interval)
                actual_word_intervals.append(
                    WordCtmInterval(
                        interval.begin,
                        interval.end,
                        self.word_mapping[self.silence_word],
                        self.pronunciation_mapping[
                            (self.silence_word, self.optional_silence_phone)
                        ],
                    )
                )
                phone_word_mapping.append(len(actual_word_intervals) - 1)
                current_word_begin = None
                current_phones = []
                continue
            if current_word_begin is None:
                current_word_begin = interval.begin
            current_phones.append(interval.label)
            try:
                cur_word = word_pronunciations[words_index]
            except IndexError:
                if self.transcription:
                    break
                else:
                    raise
            pronunciation = " ".join(cur_word[1])
            if self.position_dependent_phones:
                pronunciation = re.sub(r"_[BIES]\b", "", pronunciation)
            if current_phones == cur_word[1]:
                if (
                    pronunciation == self.oov_phone
                    and (cur_word[0], pronunciation) not in self.pronunciation_mapping
                ):
                    pron_id = self.pronunciation_mapping[(self.oov_word, pronunciation)]
                else:
                    pron_id = self.pronunciation_mapping.get((cur_word[0], pronunciation), None)
                actual_word_intervals.append(
                    WordCtmInterval(
                        current_word_begin,
                        interval.end,
                        self.word_mapping[cur_word[0]],
                        pron_id,
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

    def cleanup_g2p_intervals(
        self,
        utterance_name,
        intervals: List[CtmInterval],
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
        word_pronunciations = phones_to_prons(
            self.utterance_texts[utterance_name],
            intervals,
            self.align_lexicon_fst,
            self.word_symbol_table,
            self.phone_symbol_table,
            self.optional_silence_phone,
            clitic_marker=self.clitic_marker,
        )
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
                actual_phone_intervals.append(interval)
                actual_word_intervals.append(
                    WordCtmInterval(
                        interval.begin,
                        interval.end,
                        self.word_mapping[self.silence_word],
                        None,
                    )
                )
                phone_word_mapping.append(len(actual_word_intervals) - 1)
                current_word_begin = None
                current_phones = []
                continue
            if current_word_begin is None:
                current_word_begin = interval.begin
            current_phones.append(interval.label)
            cur_word = word_pronunciations[words_index]
            pronunciation = " ".join(cur_word[1])
            if self.position_dependent_phones:
                pronunciation = re.sub(r"_[BIES]\b", "", pronunciation)
            if current_phones == cur_word[1]:
                try:
                    if (
                        pronunciation == self.oov_phone
                        and (cur_word[0], pronunciation) not in self.pronunciation_mapping
                    ):
                        pron_id = self.pronunciation_mapping[(self.oov_word, pronunciation)]
                    else:
                        pron_id = self.pronunciation_mapping[(cur_word[0], pronunciation)]
                except KeyError:
                    pron_id = None
                try:
                    word_id = self.word_mapping[cur_word[0]]
                except KeyError:
                    word_id = cur_word[0]
                actual_word_intervals.append(
                    WordCtmInterval(
                        current_word_begin,
                        interval.end,
                        word_id,
                        pron_id,
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
        align_lexicon_paths = {}
        self.phone_symbol_table = None
        with Session(self.db_engine()) as session, mfa_open(self.log_path, "w") as log_file:
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

            self.phone_to_phone_id = {}
            ds = session.query(Phone.kaldi_label, Phone.id, Phone.mapping_id).all()
            for phone, p_id, mapping_id in ds:
                self.reversed_phone_mapping[mapping_id] = phone
                self.phone_to_phone_id[phone] = p_id
                self.phone_mapping[phone] = mapping_id

            for d in job.dictionaries:
                columns = [Utterance.id, Utterance.begin]
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
                self.utterance_begins = {}
                self.utterance_texts = {}
                for u_id, begin, text in utts:
                    self.utterance_begins[u_id] = begin
                    self.utterance_texts[u_id] = text
                if self.phone_symbol_table is None:
                    self.phone_symbol_table = pywrapfst.SymbolTable.read_text(
                        d.phone_symbol_table_path
                    )
                self.align_lexicon_fst = pynini.Fst.read(d.align_lexicon_path)
                if d.use_g2p:
                    self.word_symbol_table = pywrapfst.SymbolTable.read_text(
                        d.grapheme_symbol_table_path
                    )
                    self.align_lexicon_fst.invert()
                else:
                    self.word_symbol_table = pywrapfst.SymbolTable.read_text(d.words_symbol_path)
                self.clitic_marker = d.clitic_marker
                self.silence_word = d.silence_word
                self.oov_word = d.oov_word
                self.oov_phone = "spn"
                self.position_dependent_phones = d.position_dependent_phones
                self.optional_silence_phone = d.optional_silence_phone
                if self.transcription or self.confidence:
                    align_lexicon_paths[d.id] = d.align_lexicon_int_path
                else:
                    align_lexicon_paths[d.id] = d.align_lexicon_path
                silence_words = (
                    session.query(Word.id)
                    .filter(Word.dictionary_id == d.id)
                    .filter(Word.word_type == WordType.silence)
                )
                self.silence_words.update(x for x, in silence_words)

                words = session.query(Word.word, Word.id, Word.mapping_id).filter(
                    Word.dictionary_id == d.id
                )
                self.word_mapping = {}
                self.reversed_word_mapping = {}
                for w, w_id, m_id in words:
                    self.word_mapping[w] = w_id
                    self.reversed_word_mapping[m_id] = w
                self.pronunciation_mapping = {}
                pronunciations = (
                    session.query(Word.word, Pronunciation.pronunciation, Pronunciation.id)
                    .join(Pronunciation.word)
                    .filter(Word.dictionary_id == d.id)
                )
                for w, pron, p_id in pronunciations:
                    self.pronunciation_mapping[(w, pron)] = p_id

                lat_path = job.construct_path(
                    workflow.working_directory, "lat.carpa.rescored", "ark", d.id
                )
                if not os.path.exists(lat_path):
                    lat_path = job.construct_path(workflow.working_directory, "lat", "ark", d.id)
                ali_path = job.construct_path(workflow.working_directory, "ali", "ark", d.id)
                if self.transcription:
                    self.utterance_texts = {}
                    lat_align_proc = subprocess.Popen(
                        [
                            thirdparty_binary("lattice-align-words-lexicon"),
                            align_lexicon_paths[d.id],
                            self.model_path,
                            f"ark,s,cs:{lat_path}",
                            "ark:-",
                        ],
                        stderr=log_file,
                        stdout=subprocess.PIPE,
                        env=os.environ,
                    )
                    one_best_proc = subprocess.Popen(
                        [
                            thirdparty_binary("lattice-best-path"),
                            f"--acoustic-scale={self.score_options['acoustic_scale']}",
                            "ark,s,cs:-",
                            "ark,t:-",
                            f"ark:{ali_path}",
                        ],
                        stderr=log_file,
                        stdin=lat_align_proc.stdout,
                        stdout=subprocess.PIPE,
                        env=os.environ,
                    )
                    for line in one_best_proc.stdout:
                        line = line.strip().decode("utf8").split()
                        utt_id = int(line.pop(0).split("-")[1])
                        text = " ".join([self.reversed_word_mapping[int(x)] for x in line])
                        self.utterance_texts[utt_id] = text

                if self.confidence and os.path.exists(lat_path):
                    lat_align_proc = subprocess.Popen(
                        [
                            thirdparty_binary("lattice-align-words-lexicon"),
                            align_lexicon_paths[d.id],
                            self.model_path,
                            f"ark,s,cs:{lat_path}",
                            "ark:-",
                        ],
                        stderr=log_file,
                        stdout=subprocess.PIPE,
                        env=os.environ,
                    )
                    phone_lat_proc = subprocess.Popen(
                        [
                            thirdparty_binary("lattice-to-phone-lattice"),
                            "--replace-words=true",
                            self.model_path,
                            "ark,s,cs:-",
                            "ark:-",
                        ],
                        stderr=log_file,
                        stdin=lat_align_proc.stdout,
                        stdout=subprocess.PIPE,
                        env=os.environ,
                    )
                    ctm_proc = subprocess.Popen(
                        [
                            thirdparty_binary("lattice-to-ctm-conf"),
                            f"--acoustic-scale={self.score_options['acoustic_scale']}",
                            "ark,s,cs:-",
                            "-",
                        ],
                        stderr=log_file,
                        stdin=phone_lat_proc.stdout,
                        stdout=subprocess.PIPE,
                        env=os.environ,
                        encoding="utf8",
                    )
                    for utterance, intervals in parse_ctm_output(
                        ctm_proc, self.reversed_phone_mapping
                    ):
                        try:
                            (
                                word_intervals,
                                phone_intervals,
                                phone_word_mapping,
                            ) = self.cleanup_intervals(utterance, intervals)
                        except pywrapfst.FstOpError:
                            log_file.write(f"Error for {utterance}\n")
                            log_file.write(f"{self.utterance_texts[utterance]}\n")
                            log_file.write(f"{' '.join(x.label for x in intervals)}\n")
                            log_file.flush()
                            continue
                        yield utterance, word_intervals, phone_intervals, phone_word_mapping

                    self.check_call(ctm_proc)
                else:
                    ctm_proc = subprocess.Popen(
                        [
                            thirdparty_binary("ali-to-phones"),
                            "--ctm-output",
                            f"--frame-shift={self.frame_shift}",
                            self.model_path,
                            f"ark,s,cs:{ali_path}",
                            "-",
                        ],
                        stderr=log_file,
                        stdout=subprocess.PIPE,
                        env=os.environ,
                        encoding="utf8",
                    )
                    for utterance, intervals in parse_ctm_output(
                        ctm_proc, self.reversed_phone_mapping
                    ):
                        if not d.use_g2p:

                            (
                                word_intervals,
                                phone_intervals,
                                phone_word_mapping,
                            ) = self.cleanup_intervals(utterance, intervals)
                        else:
                            try:
                                (
                                    word_intervals,
                                    phone_intervals,
                                    phone_word_mapping,
                                ) = self.cleanup_g2p_intervals(utterance, intervals)
                            except pywrapfst.FstOpError:
                                continue
                        yield utterance, word_intervals, phone_intervals, phone_word_mapping
                    self.check_call(ctm_proc)


def construct_output_tiers(
    session: Session,
    file_id: int,
    workflow: CorpusWorkflow,
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
    data = {}
    for utt in utterances:
        word_intervals = (
            session.query(WordInterval, Word)
            .join(WordInterval.word)
            .filter(WordInterval.utterance_id == utt.id)
            .filter(WordInterval.workflow_id == workflow.id)
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
                if (
                    w.word_type is WordType.oov
                    and workflow.workflow_type is WorkflowType.alignment
                ):
                    label = actual_words[i]
                if (
                    data[utt.speaker.name]["words"]
                    and clitic_marker
                    and (
                        data[utt.speaker.name]["words"][-1].label.endswith(clitic_marker)
                        or label.startswith(clitic_marker)
                    )
                ):
                    data[utt.speaker.name]["words"][-1].end = wi.end
                    data[utt.speaker.name]["words"][-1].label += label

                    for pi in sorted(wi.phone_intervals, key=lambda x: x.begin):
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
    relative_path: Path,
    output_directory: Path,
    input_path: Path = None,
    output_format: str = TextgridFormats.SHORT_TEXTGRID,
) -> Path:
    """
    Construct an output path

    Returns
    -------
    Path
        Output path
    """
    if isinstance(output_directory, str):
        output_directory = Path(output_directory)
    if output_format.upper() == "LAB":
        extension = ".lab"
    elif output_format.upper() == "JSON":
        extension = ".json"
    elif output_format.upper() == "CSV":
        extension = ".csv"
    else:
        extension = ".TextGrid"
    if relative_path:
        relative = output_directory.joinpath(relative_path)
    else:
        relative = output_directory
    output_path = relative.joinpath(name + extension)
    if output_path == input_path:
        output_path = relative.joinpath(name + "_aligned" + extension)
    os.makedirs(relative, exist_ok=True)
    relative.mkdir(parents=True, exist_ok=True)
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
        self.cleanup_textgrids = arguments.cleanup_textgrids
        self.clitic_marker = arguments.clitic_marker
        self.exported_file_count = exported_file_count

    def run(self) -> None:
        """Run the exporter function"""
        db_engine = sqlalchemy.create_engine(
            self.db_string,
            poolclass=sqlalchemy.NullPool,
            pool_reset_on_return=None,
            logging_name=f"{type(self).__name__}_engine",
            isolation_level="AUTOCOMMIT",
        ).execution_options(logging_token=f"{type(self).__name__}_engine")
        with mfa_open(self.log_path, "w") as log_file, Session(db_engine) as session:
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
                        workflow,
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
            log_file.write("Done!\n")
