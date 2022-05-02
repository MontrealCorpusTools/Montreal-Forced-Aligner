"""
Alignment multiprocessing functions
-----------------------------------

"""
from __future__ import annotations

import multiprocessing as mp
import os
import re
import subprocess
import sys
import traceback
import typing
from queue import Empty
from typing import TYPE_CHECKING, Dict, List, Union

import sqlalchemy.engine
from sqlalchemy.orm import Session, joinedload, load_only, selectinload

from montreal_forced_aligner.data import (
    CtmInterval,
    MfaArguments,
    PronunciationProbabilityCounter,
    TextgridFormats,
)
from montreal_forced_aligner.db import Dictionary, File, Phone, Speaker, Utterance, Word
from montreal_forced_aligner.helper import split_phone_position
from montreal_forced_aligner.textgrid import export_textgrid, process_ctm_line
from montreal_forced_aligner.utils import Counter, KaldiFunction, Stopped, thirdparty_binary

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
    """Arguments for :func:`~montreal_forced_aligner.alignment.multiprocessing.GeneratePronunciationsFunction`"""

    text_int_paths: Dict[int, str]
    ali_paths: Dict[int, str]
    model_path: str


@dataclass
class AlignmentExtractionArguments(MfaArguments):
    """Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.AlignmentExtractionFunction`"""

    model_path: str
    frame_shift: float
    cleanup_textgrids: bool
    ali_paths: Dict[int, str]
    text_int_paths: Dict[int, str]


@dataclass
class ExportTextGridArguments(MfaArguments):
    """Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.ExportTextGridProcessWorker`"""

    frame_shift: int
    output_directory: str
    output_format: str


@dataclass
class CompileInformationArguments(MfaArguments):
    """Arguments for :func:`~montreal_forced_aligner.alignment.multiprocessing.compile_information_func`"""

    align_log_path: str


@dataclass
class CompileTrainGraphsArguments(MfaArguments):
    """Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.CompileTrainGraphsFunction`"""

    dictionaries: List[int]
    tree_path: str
    model_path: str
    text_int_paths: Dict[int, str]
    fst_scp_paths: Dict[int, str]


@dataclass
class AlignArguments(MfaArguments):
    """Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.AlignFunction`"""

    dictionaries: List[int]
    fst_scp_paths: Dict[int, str]
    feature_strings: Dict[int, str]
    model_path: str
    ali_paths: Dict[int, str]
    align_options: MetaDict


@dataclass
class AccStatsArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.AccStatsFunction`
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
        self.fst_scp_paths = args.fst_scp_paths

    def run(self):
        """Run the function"""
        db_engine = sqlalchemy.create_engine(f"sqlite:///{self.db_path}?mode=ro&nolock=1")
        with open(self.log_path, "w", encoding="utf8") as log_file, Session(db_engine) as session:
            dictionaries = (
                session.query(Dictionary)
                .join(Dictionary.speakers)
                .filter(Speaker.job_id == self.job_name)
                .distinct()
            )
            for d in dictionaries:
                fst_scp_path = self.fst_scp_paths[d.id]
                fst_ark_path = fst_scp_path.replace(".scp", ".ark")
                text_path = self.text_int_paths[d.id]
                proc = subprocess.Popen(
                    [
                        thirdparty_binary("compile-train-graphs"),
                        f"--read-disambig-syms={d.disambiguation_symbols_int_path}",
                        self.tree_path,
                        self.model_path,
                        d.lexicon_fst_path,
                        f"ark:{text_path}",
                        f"ark,scp:{fst_ark_path},{fst_scp_path}",
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

    def run(self):
        """Run the function"""
        with open(self.log_path, "w", encoding="utf8") as log_file:
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
        self.fst_scp_paths = args.fst_scp_paths
        self.feature_strings = args.feature_strings
        self.model_path = args.model_path
        self.ali_paths = args.ali_paths
        self.align_options = args.align_options

    def run(self):
        """Run the function"""
        with open(self.log_path, "w", encoding="utf8") as log_file:
            for dict_id in self.dictionaries:
                feature_string = self.feature_strings[dict_id]
                fst_path = self.fst_scp_paths[dict_id]
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
                    f"scp:{fst_path}",
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
        self.reversed_phone_mapping = {}
        self.word_boundary_int_paths = {}
        self.reversed_word_mapping = {}

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
                if word == self.silence_word:
                    counter.silence_before_counts[w_p] += 1
                else:
                    counter.non_silence_before_counts[w_p] += 1
            silence_check = w_p[0] == self.silence_word
            if not silence_check:
                counter.word_pronunciation_counts[w_p[0]][w_p[1]] += 1
                if i != len(word_pronunciations) - 1:
                    word = word_pronunciations[i + 1][0]
                    if word == self.silence_word:
                        counter.silence_following_counts[w_p] += 1
                        if i != len(word_pronunciations) - 2:
                            next_w_p = word_pronunciations[i + 2]
                            counter.ngram_counts[w_p, next_w_p]["silence"] += 1
                    else:
                        next_w_p = word_pronunciations[i + 1]
                        counter.non_silence_following_counts[w_p] += 1
                        counter.ngram_counts[w_p, next_w_p]["non_silence"] += 1
        return counter

    def run(self):
        """Run the function"""
        db_engine = sqlalchemy.create_engine(f"sqlite:///{self.db_path}?mode=ro&nolock=1")
        with Session(db_engine) as session:
            ds = session.query(Dictionary).options(
                selectinload(Dictionary.words),
                load_only(
                    Dictionary.position_dependent_phones,
                    Dictionary.clitic_marker,
                    Dictionary.silence_word,
                    Dictionary.oov_word,
                    Dictionary.id,
                    Dictionary.optional_silence_phone,
                    Dictionary.root_temp_directory,
                ),
            )
            for d in ds:
                if d.id not in self.text_int_paths:
                    continue
                self.position_dependent_phones = d.position_dependent_phones
                self.clitic_marker = d.clitic_marker
                self.silence_word = d.silence_word
                self.oov_word = d.oov_word
                self.optional_silence_phone = d.optional_silence_phone
                self.word_boundary_int_paths[d.id] = d.word_boundary_int_path
                self.reversed_word_mapping[d.id] = {}
                for w in d.words:
                    self.reversed_word_mapping[d.id][w.mapping_id] = w.word
            phones = session.query(Phone.phone, Phone.mapping_id)
            for phone, mapping_id in phones:
                self.reversed_phone_mapping[mapping_id] = phone
        with open(self.log_path, "w", encoding="utf8") as log_file:
            for dict_id in self.text_int_paths.keys():
                current_utterance = None
                word_pronunciations = []
                text_int_path = self.text_int_paths[dict_id]
                word_boundary_path = self.word_boundary_int_paths[dict_id]
                ali_path = self.ali_paths[dict_id]
                if not os.path.exists(ali_path):
                    continue
                lin_proc = subprocess.Popen(
                    [
                        thirdparty_binary("linear-to-nbest"),
                        f"ark:{ali_path}",
                        f"ark:{text_int_path}",
                        "",
                        "",
                        "ark:-",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=os.environ,
                )
                align_proc = subprocess.Popen(
                    [
                        thirdparty_binary("lattice-align-words"),
                        word_boundary_path,
                        self.model_path,
                        "ark:-",
                        "ark:-",
                    ],
                    stdin=lin_proc.stdout,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=os.environ,
                )

                prons_proc = subprocess.Popen(
                    [thirdparty_binary("nbest-to-prons"), self.model_path, "ark:-", "-"],
                    stdin=align_proc.stdout,
                    stderr=log_file,
                    encoding="utf8",
                    stdout=subprocess.PIPE,
                    env=os.environ,
                )
                for line in prons_proc.stdout:
                    line = line.strip().split()
                    utt = line[0]
                    if utt != current_utterance and current_utterance is not None:
                        log_file.write(f"{current_utterance}\t{word_pronunciations}\n")
                        yield dict_id, self._process_pronunciations(word_pronunciations)
                        word_pronunciations = []
                    current_utterance = utt
                    pron = [int(x) for x in line[4:]]
                    word = self.reversed_word_mapping[dict_id][int(line[3])]
                    if self.position_dependent_phones:
                        pron = " ".join(
                            split_phone_position(self.reversed_phone_mapping[x])[0] for x in pron
                        )
                    else:
                        pron = " ".join(self.reversed_phone_mapping[x] for x in pron)
                    word_pronunciations.append((word, pron))
                if word_pronunciations:
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
    with open(arguments.log_path, "w", encoding="utf8"), open(
        align_log_path, "r", encoding="utf8"
    ) as f:
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
        self.cleanup_textgrids = args.cleanup_textgrids
        self.utterance_texts = {}
        self.utterance_begins = {}
        self.word_boundary_int_paths = {}
        self.reversed_phone_mapping = {}
        self.words = {}
        self.silence_words = set()

    def cleanup_intervals(self, utterance_name: int, dict_id: int, intervals: List[CtmInterval]):
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
        utterance_name = utterance_name
        utterance_begin = self.utterance_begins[utterance_name]
        current_word_begin = None
        words = self.utterance_texts[utterance_name]
        words_index = 0
        current_phones = []
        for interval in intervals:
            interval.begin += utterance_begin
            interval.end += utterance_begin
            phone_label = self.reversed_phone_mapping[int(interval.label)]
            if phone_label == self.optional_silence_phone:
                if words_index < len(words) and words[words_index] in self.silence_words:
                    interval.label = phone_label
                elif self.cleanup_textgrids:
                    continue
                else:
                    interval.label = phone_label
                    actual_phone_intervals.append(interval)
                    continue
            if self.position_dependent_phones and "_" in phone_label:
                phone, position = split_phone_position(phone_label)
                if position in {"B", "S"}:
                    current_word_begin = interval.begin
                if position in {"E", "S"}:
                    if (
                        self.cleanup_textgrids
                        and actual_word_intervals
                        and self.clitic_marker
                        and (
                            actual_word_intervals[-1].label.endswith(self.clitic_marker)
                            or words[words_index].endswith(self.clitic_marker)
                        )
                    ):
                        actual_word_intervals[-1].end = interval.end
                        actual_word_intervals[-1].label += words[words_index]
                    else:
                        actual_word_intervals.append(
                            CtmInterval(
                                current_word_begin,
                                interval.end,
                                words[words_index],
                                utterance_name,
                            )
                        )
                    words_index += 1
                    current_word_begin = None
                interval.label = phone
            else:
                if current_word_begin is None:
                    current_word_begin = interval.begin
                current_phones.append(phone_label)
                cur_word = words[words_index]
                if cur_word not in self.words[dict_id]:
                    cur_word = self.oov_word
                if tuple(current_phones) in self.words[dict_id][cur_word]:
                    actual_word_intervals.append(
                        CtmInterval(
                            current_word_begin, interval.end, words[words_index], utterance_name
                        )
                    )
                    current_word_begin = None
                    current_phones = []
                    words_index += 1
            actual_phone_intervals.append(interval)
        return actual_word_intervals, actual_phone_intervals

    def run(self):
        """Run the function"""
        db_engine = sqlalchemy.create_engine(f"sqlite:///{self.db_path}?mode=ro&nolock=1")
        with Session(db_engine) as session:
            for dict_id in self.ali_paths.keys():
                d = (
                    session.query(Dictionary)
                    .options(
                        selectinload(Dictionary.words).selectinload(Word.pronunciations),
                        load_only(
                            Dictionary.position_dependent_phones,
                            Dictionary.clitic_marker,
                            Dictionary.silence_word,
                            Dictionary.oov_word,
                            Dictionary.root_temp_directory,
                            Dictionary.optional_silence_phone,
                        ),
                    )
                    .get(dict_id)
                )

                self.position_dependent_phones = d.position_dependent_phones
                self.clitic_marker = d.clitic_marker
                self.silence_word = d.silence_word
                self.oov_word = d.oov_word
                self.optional_silence_phone = d.optional_silence_phone
                self.word_boundary_int_paths[dict_id] = d.word_boundary_int_path

                self.words[dict_id] = {}
                for w in d.words:
                    self.words[dict_id][w.word] = set()
                    for pron in w.pronunciations:
                        if pron == self.optional_silence_phone:
                            self.silence_words.add(w.word)
                        self.words[dict_id][w.word].add(tuple(pron.pronunciation.split(" ")))
                utts = (
                    session.query(Utterance)
                    .join(Utterance.speaker)
                    .options(load_only(Utterance.id, Utterance.normalized_text, Utterance.begin))
                    .filter(Speaker.job_id == self.job_name)
                )
                for utt in utts:
                    self.utterance_texts[utt.id] = utt.normalized_text.split()
                    self.utterance_begins[utt.id] = utt.begin
                ds = session.query(Phone.phone, Phone.mapping_id).all()
                for phone, mapping_id in ds:
                    self.reversed_phone_mapping[mapping_id] = phone
        with open(self.log_path, "w", encoding="utf8") as log_file:
            for dict_id in self.ali_paths.keys():
                cur_utt = None
                intervals = []
                ali_path = self.ali_paths[dict_id]
                text_int_path = self.text_int_paths[dict_id]
                word_boundary_int_path = self.word_boundary_int_paths[dict_id]
                lin_proc = subprocess.Popen(
                    [
                        thirdparty_binary("linear-to-nbest"),
                        f"ark:{ali_path}",
                        f"ark:{text_int_path}",
                        "",
                        "",
                        "ark:-",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=os.environ,
                )
                align_words_proc = subprocess.Popen(
                    [
                        thirdparty_binary("lattice-align-words"),
                        word_boundary_int_path,
                        self.model_path,
                        "ark:-",
                        "ark:-",
                    ],
                    stdin=lin_proc.stdout,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=os.environ,
                )
                phone_proc = subprocess.Popen(
                    [
                        thirdparty_binary("lattice-to-phone-lattice"),
                        self.model_path,
                        "ark:-",
                        "ark:-",
                    ],
                    stdout=subprocess.PIPE,
                    stdin=align_words_proc.stdout,
                    stderr=log_file,
                    env=os.environ,
                )
                nbest_proc = subprocess.Popen(
                    [
                        thirdparty_binary("nbest-to-ctm"),
                        "--print-args=false",
                        f"--frame-shift={self.frame_shift}",
                        "ark:-",
                        "-",
                    ],
                    stdin=phone_proc.stdout,
                    stderr=log_file,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                    encoding="utf8",
                )
                for line in nbest_proc.stdout:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        interval = process_ctm_line(line)
                    except ValueError:
                        continue
                    if cur_utt is None:
                        cur_utt = interval.utterance
                    if cur_utt != interval.utterance:
                        word_intervals, phone_intervals = self.cleanup_intervals(
                            cur_utt, dict_id, intervals
                        )
                        yield cur_utt, word_intervals, phone_intervals
                        intervals = []
                        cur_utt = interval.utterance
                    intervals.append(interval)
                self.check_call(nbest_proc)
            if intervals:
                word_intervals, phone_intervals = self.cleanup_intervals(
                    cur_utt, dict_id, intervals
                )
                yield cur_utt, word_intervals, phone_intervals


def construct_output_tiers(
    session: Session, file: File
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
    data = {}
    for utt in file.utterances:
        if utt.speaker.name not in data:
            data[utt.speaker.name] = {"words": [], "phones": []}
        for wi in utt.word_intervals:
            data[utt.speaker.name]["words"].append(CtmInterval(wi.begin, wi.end, wi.label, utt.id))

        for pi in utt.phone_intervals:
            data[utt.speaker.name]["phones"].append(
                CtmInterval(pi.begin, pi.end, pi.label, utt.id)
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
    else:
        extension = ".TextGrid"
    if relative_path:
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
        db_path: str,
        for_write_queue: mp.Queue,
        return_queue: mp.Queue,
        stopped: Stopped,
        finished_adding: Stopped,
        arguments: ExportTextGridArguments,
        exported_file_count: Counter,
    ):
        mp.Process.__init__(self)
        self.db_path = db_path
        self.for_write_queue = for_write_queue
        self.return_queue = return_queue
        self.stopped = stopped
        self.finished_adding = finished_adding
        self.finished_processing = Stopped()

        self.output_directory = arguments.output_directory
        self.output_format = arguments.output_format
        self.frame_shift = arguments.frame_shift
        self.log_path = arguments.log_path
        self.exported_file_count = exported_file_count

    def run(self) -> None:
        """Run the exporter function"""
        db_engine = sqlalchemy.create_engine(f"sqlite:///{self.db_path}?mode=ro&nolock=1")
        with open(self.log_path, "w", encoding="utf8") as log_file, Session(db_engine) as session:

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
                    utterances = (
                        session.query(Utterance)
                        .options(
                            joinedload(Utterance.speaker, innerjoin=True).load_only(Speaker.name),
                            selectinload(Utterance.phone_intervals),
                            selectinload(Utterance.word_intervals),
                        )
                        .filter(Utterance.file_id == file_id)
                    )
                    data = {}
                    for utt in utterances:
                        if utt.speaker.name not in data:
                            data[utt.speaker.name] = {"words": [], "phones": []}
                        for wi in utt.word_intervals:
                            data[utt.speaker.name]["words"].append(
                                CtmInterval(wi.begin, wi.end, wi.label, utt.id)
                            )

                        for pi in utt.phone_intervals:
                            data[utt.speaker.name]["phones"].append(
                                CtmInterval(pi.begin, pi.end, pi.label, utt.id)
                            )
                    export_textgrid(
                        data, output_path, duration, self.frame_shift, self.output_format
                    )
                    self.return_queue.put(1)
                except Exception:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    log_file.write(
                        f"Error writing to {output_path}: \n\n{self.textgrid_errors[output_path]}\n"
                    )
                    self.stopped.stop()
                    self.return_queue.put(
                        (
                            output_path,
                            "\n".join(
                                traceback.format_exception(exc_type, exc_value, exc_traceback)
                            ),
                        )
                    )
                    raise
            log_file.write("Done!\n")
