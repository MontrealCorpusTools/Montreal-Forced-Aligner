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
from queue import Empty
from typing import TYPE_CHECKING, NamedTuple, Union

from montreal_forced_aligner.textgrid import (
    CtmInterval,
    export_textgrid,
    generate_tiers,
    parse_from_phone,
    parse_from_word,
    parse_from_word_no_cleanup,
    process_ctm_line,
)
from montreal_forced_aligner.utils import Stopped, thirdparty_binary

if TYPE_CHECKING:
    from montreal_forced_aligner.abc import CtmErrorDict, MetaDict, ReversedMappingType
    from montreal_forced_aligner.corpus.classes import File, Speaker, Utterance
    from montreal_forced_aligner.dictionary import DictionaryData


queue_polling_timeout = 1

__all__ = [
    "PhoneCtmProcessWorker",
    "CleanupWordCtmProcessWorker",
    "NoCleanupWordCtmProcessWorker",
    "CombineProcessWorker",
    "ExportPreparationProcessWorker",
    "ExportTextGridProcessWorker",
    "align_func",
    "ali_to_ctm_func",
    "compile_information_func",
    "compile_train_graphs_func",
]


class AliToCtmArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.alignment.multiprocessing.ali_to_ctm_func`"""

    log_path: str
    dictionaries: list[str]
    ali_paths: dict[str, str]
    text_int_paths: dict[str, str]
    word_boundary_int_paths: dict[str, str]
    frame_shift: float
    model_path: str
    ctm_paths: dict[str, str]
    word_mode: bool


class CleanupWordCtmArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.CleanupWordCtmProcessWorker`"""

    ctm_paths: dict[str, str]
    dictionaries: list[str]
    utterances: dict[str, dict[str, Utterance]]
    dictionary_data: dict[str, DictionaryData]


class NoCleanupWordCtmArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.NoCleanupWordCtmProcessWorker`"""

    ctm_paths: dict[str, str]
    dictionaries: list[str]
    utterances: dict[str, dict[str, Utterance]]
    dictionary_data: dict[str, DictionaryData]


class PhoneCtmArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.PhoneCtmProcessWorker`"""

    ctm_paths: dict[str, str]
    dictionaries: list[str]
    utterances: dict[str, dict[str, Utterance]]
    reversed_phone_mappings: dict[str, ReversedMappingType]
    positions: dict[str, list[str]]


class CombineCtmArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.CombineProcessWorker`"""

    dictionaries: list[str]
    files: dict[str, File]
    speakers: dict[str, Speaker]
    dictionary_data: dict[str, DictionaryData]
    cleanup_textgrids: bool


class ExportTextGridArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.ExportTextGridProcessWorker`"""

    files: dict[str, File]
    frame_shift: int
    output_directory: str
    backup_output_directory: str


class CompileInformationArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.alignment.multiprocessing.compile_information_func`"""

    align_log_paths: str


class CompileTrainGraphsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.alignment.multiprocessing.compile_train_graphs_func`"""

    log_path: str
    dictionaries: list[str]
    tree_path: str
    model_path: str
    text_int_paths: dict[str, str]
    disambig_paths: dict[str, str]
    lexicon_fst_paths: dict[str, str]
    fst_scp_paths: dict[str, str]


class AlignArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.alignment.multiprocessing.align_func`"""

    log_path: str
    dictionaries: list[str]
    fst_scp_paths: dict[str, str]
    feature_strings: dict[str, str]
    model_path: str
    ali_paths: dict[str, str]
    align_options: MetaDict


def compile_train_graphs_func(
    log_path: str,
    dictionaries: list[str],
    tree_path: str,
    model_path: str,
    text_int_paths: dict[str, str],
    disambig_path: str,
    lexicon_fst_paths: dict[str, str],
    fst_scp_paths: dict[str, str],
) -> None:
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
    log_path: str
        Path to save log output
    dictionaries: list[str]
        List of dictionary names
    tree_path: str
        Path to the acoustic model tree file
    model_path: str
        Path to the acoustic model file
    text_int_paths: dict[str, str]
        Dictionary of text int files per dictionary name
    disambig_path: str
        Disambiguation symbol int file
    lexicon_fst_paths: dict[str, str]
        Dictionary of L.fst files per dictionary name
    fst_scp_paths: dict[str, str]
        Dictionary of utterance FST scp files per dictionary name
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            fst_scp_path = fst_scp_paths[dict_name]
            fst_ark_path = fst_scp_path.replace(".scp", ".ark")
            text_path = text_int_paths[dict_name]
            log_file.write(f"{dict_name}\t{fst_scp_path}\t{fst_ark_path}\t{text_path}\n\n")
            log_file.flush()
            proc = subprocess.Popen(
                [
                    thirdparty_binary("compile-train-graphs"),
                    f"--read-disambig-syms={disambig_path}",
                    tree_path,
                    model_path,
                    lexicon_fst_paths[dict_name],
                    f"ark:{text_path}",
                    f"ark,scp:{fst_ark_path},{fst_scp_path}",
                ],
                stderr=log_file,
                env=os.environ,
            )
            proc.communicate()


def align_func(
    log_path: str,
    dictionaries: list[str],
    fst_scp_paths: dict[str, str],
    feature_strings: dict[str, str],
    model_path: str,
    ali_paths: dict[str, str],
    align_options: MetaDict,
):
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
    log_path: str
        Path to save log output
    dictionaries: list[str]
        List of dictionary names
    fst_scp_paths: dict[str, str]
        Dictionary of FST scp file paths per dictionary name
    feature_strings: dict[str, str]
        Dictionary of feature strings per dictionary name
    model_path: str
        Path to the acoustic model file
    ali_paths: dict[str, str]
        Dictionary of alignment archives per dictionary name
    align_options: dict[str, Any]
        Options for alignment
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            fst_path = fst_scp_paths[dict_name]
            ali_path = ali_paths[dict_name]
            com = [
                thirdparty_binary("gmm-align-compiled"),
                f"--transition-scale={align_options['transition_scale']}",
                f"--acoustic-scale={align_options['acoustic_scale']}",
                f"--self-loop-scale={align_options['self_loop_scale']}",
                f"--beam={align_options['beam']}",
                f"--retry-beam={align_options['retry_beam']}",
                "--careful=false",
                "-",
                f"scp:{fst_path}",
                feature_string,
                f"ark:{ali_path}",
            ]

            boost_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-boost-silence"),
                    f"--boost={align_options['boost_silence']}",
                    align_options["optional_silence_csl"],
                    model_path,
                    "-",
                ],
                stderr=log_file,
                stdout=subprocess.PIPE,
                env=os.environ,
            )
            align_proc = subprocess.Popen(
                com, stderr=log_file, stdin=boost_proc.stdout, env=os.environ
            )
            align_proc.communicate()


def compile_information_func(align_log_path: str) -> dict[str, Union[list[str], float, int]]:
    """
    Multiprocessing function for compiling information about alignment

    See Also
    --------
    :meth:`.AlignMixin.compile_information`
        Main function that calls this function in parallel

    Parameters
    ----------
    align_log_path: str
        Log path for alignment

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
    with open(align_log_path, "r", encoding="utf8") as f:
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


def ali_to_ctm_func(
    log_path: str,
    dictionaries: list[str],
    ali_paths: dict[str, str],
    text_int_paths: dict[str, str],
    word_boundary_int_paths: dict[str, str],
    frame_shift: float,
    model_path: str,
    ctm_paths: dict[str, str],
    word_mode: bool,
) -> None:
    """
    Multiprocessing function to convert alignment archives into CTM files

    See Also
    --------
    :meth:`.CorpusAligner.ctms_to_textgrids_mp`
        Main function that calls this function in parallel
    :meth:`.CorpusAligner.ali_to_word_ctm_arguments`
        Job method for generating arguments for this function
    :meth:`.CorpusAligner.ali_to_phone_ctm_arguments`
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

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: list[str]
        List of dictionary names
    ali_paths: dict[str, str]
        Dictionary of alignment archives per dictionary name
    text_int_paths: dict[str, str]
        Dictionary of text int files per dictionary name
    word_boundary_int_paths: dict[str, str]
        Dictionary of word boundary int files per dictionary name
    frame_shift: float
        Frame shift of feature generation in seconds
    model_path: str
        Path to the acoustic model file
    ctm_paths: dict[str, str]
        Dictionary of CTM files per dictionary name
    word_mode: bool
        Flag for whether to parse words or phones
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            ali_path = ali_paths[dict_name]
            text_int_path = text_int_paths[dict_name]
            ctm_path = ctm_paths[dict_name]
            word_boundary_int_path = word_boundary_int_paths[dict_name]
            if os.path.exists(ctm_path):
                return
            lin_proc = subprocess.Popen(
                [
                    thirdparty_binary("linear-to-nbest"),
                    "ark:" + ali_path,
                    "ark:" + text_int_path,
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
                    model_path,
                    "ark:-",
                    "ark:-",
                ],
                stdin=lin_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            if word_mode:
                nbest_proc = subprocess.Popen(
                    [
                        thirdparty_binary("nbest-to-ctm"),
                        f"--frame-shift={frame_shift}",
                        "ark:-",
                        ctm_path,
                    ],
                    stderr=log_file,
                    stdin=align_words_proc.stdout,
                    env=os.environ,
                )
            else:
                phone_proc = subprocess.Popen(
                    [thirdparty_binary("lattice-to-phone-lattice"), model_path, "ark:-", "ark:-"],
                    stdout=subprocess.PIPE,
                    stdin=align_words_proc.stdout,
                    stderr=log_file,
                    env=os.environ,
                )
                nbest_proc = subprocess.Popen(
                    [
                        thirdparty_binary("nbest-to-ctm"),
                        f"--frame-shift={frame_shift}",
                        "ark:-",
                        ctm_path,
                    ],
                    stdin=phone_proc.stdout,
                    stderr=log_file,
                    env=os.environ,
                )
            nbest_proc.communicate()


class NoCleanupWordCtmProcessWorker(mp.Process):
    """
    Multiprocessing worker for loading word CTM files without any clean up

    See Also
    --------
    :meth:`.CorpusAligner.ctms_to_textgrids_mp`
        Main function that runs this worker in parallel

    Parameters
    ----------
    job_name: int
        Job name
    to_process_queue: :class:`~multiprocessing.Queue`
        Return queue of jobs for later workers to process
    stopped: :class:`~montreal_forced_aligner.utils.Stopped`
        Stop check for processing
    error_catching: dict[tuple[str, int], str]
        Dictionary for storing errors encountered
    arguments: :class:`~montreal_forced_aligner.alignment.multiprocessing.NoCleanupWordCtmArguments`
        Arguments to pass to the CTM processing function
    """

    def __init__(
        self,
        job_name: int,
        to_process_queue: mp.Queue,
        stopped: Stopped,
        error_catching: CtmErrorDict,
        arguments: NoCleanupWordCtmArguments,
    ):
        mp.Process.__init__(self)
        self.job_name = job_name
        self.dictionaries = arguments.dictionaries
        self.ctm_paths = arguments.ctm_paths
        self.to_process_queue = to_process_queue
        self.stopped = stopped
        self.error_catching = error_catching

        # Corpus information
        self.utterances = arguments.utterances

        # Dictionary information
        self.dictionary_data = arguments.dictionary_data

    def run(self) -> None:
        """
        Run the word processing with no clean up
        """
        current_file_data = {}

        def process_current(cur_utt: Utterance, current_labels: list[CtmInterval]):
            """Process current stack of intervals"""
            actual_labels = parse_from_word_no_cleanup(
                current_labels, self.dictionary_data[dict_name].reversed_words_mapping
            )
            current_file_data[cur_utt.name] = actual_labels

        def process_current_file(cur_file: str):
            """Process current file and add to return queue"""
            self.to_process_queue.put(("word", cur_file, current_file_data))

        cur_utt = None
        cur_file = None
        utt_begin = 0
        current_labels = []
        try:
            for dict_name in self.dictionaries:
                with open(self.ctm_paths[dict_name], "r") as word_file:
                    for line in word_file:
                        line = line.strip()
                        if not line:
                            continue
                        interval = process_ctm_line(line)
                        utt = interval.utterance
                        if cur_utt is None:
                            cur_utt = self.utterances[dict_name][utt]
                            utt_begin = cur_utt.begin
                            cur_file = cur_utt.file_name

                        if utt != cur_utt:
                            process_current(cur_utt, current_labels)
                            cur_utt = self.utterances[dict_name][utt]
                            file_name = cur_utt.file_name
                            if file_name != cur_file:
                                process_current_file(cur_file)
                                current_file_data = {}
                                cur_file = file_name
                            current_labels = []
                        if utt_begin:
                            interval.shift_times(utt_begin)
                        current_labels.append(interval)
                if current_labels:
                    process_current(cur_utt, current_labels)
                    process_current_file(cur_file)
        except Exception:
            self.stopped.stop()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.error_catching[("word", self.job_name)] = "\n".join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)
            )


class CleanupWordCtmProcessWorker(mp.Process):
    """
    Multiprocessing worker for loading word CTM files with cleaning up MFA-internal modifications

    See Also
    --------
    :meth:`.CorpusAligner.ctms_to_textgrids_mp`
        Main function that runs this worker in parallel

    Parameters
    ----------
    job_name: int
        Job name
    to_process_queue: :class:`~multiprocessing.Queue`
        Return queue of jobs for later workers to process
    stopped: :class:`~montreal_forced_aligner.utils.Stopped`
        Stop check for processing
    error_catching: dict[tuple[str, int], str]
        Dictionary for storing errors encountered
    arguments: :class:`~montreal_forced_aligner.alignment.multiprocessing.CleanupWordCtmArguments`
        Arguments to pass to the CTM processing function
    """

    def __init__(
        self,
        job_name: int,
        to_process_queue: mp.Queue,
        stopped: Stopped,
        error_catching: CtmErrorDict,
        arguments: CleanupWordCtmArguments,
    ):
        mp.Process.__init__(self)
        self.job_name = job_name
        self.dictionaries = arguments.dictionaries
        self.ctm_paths = arguments.ctm_paths
        self.to_process_queue = to_process_queue
        self.stopped = stopped
        self.error_catching = error_catching

        # Corpus information
        self.utterances = arguments.utterances

        # Dictionary information
        self.dictionary_data = arguments.dictionary_data

    def run(self) -> None:
        """
        Run the word processing with clean up
        """
        current_file_data = {}

        def process_current(cur_utt: Utterance, current_labels: list[CtmInterval]) -> None:
            """Process current stack of intervals"""
            text = cur_utt.text.split()
            actual_labels = parse_from_word(current_labels, text, self.dictionary_data[dict_name])

            current_file_data[cur_utt.name] = actual_labels

        def process_current_file(cur_file: str) -> None:
            """Process current file and add to return queue"""
            self.to_process_queue.put(("word", cur_file, current_file_data))

        cur_utt = None
        cur_file = None
        utt_begin = 0
        current_labels = []
        try:
            for dict_name in self.dictionaries:
                ctm_path = self.ctm_paths[dict_name]
                with open(ctm_path, "r") as word_file:
                    for line in word_file:
                        line = line.strip()
                        if not line:
                            continue
                        interval = process_ctm_line(line)
                        utt = interval.utterance
                        if cur_utt is None:
                            cur_utt = self.utterances[dict_name][utt]
                            utt_begin = cur_utt.begin
                            cur_file = cur_utt.file_name

                        if utt != cur_utt:
                            process_current(cur_utt, current_labels)
                            cur_utt = self.utterances[dict_name][utt]
                            utt_begin = cur_utt.begin
                            file_name = cur_utt.file_name
                            if file_name != cur_file:
                                process_current_file(cur_file)
                                current_file_data = {}
                                cur_file = file_name
                            current_labels = []
                        if utt_begin:
                            interval.shift_times(utt_begin)
                        current_labels.append(interval)
                if current_labels:
                    process_current(cur_utt, current_labels)
                    process_current_file(cur_file)
        except Exception:
            self.stopped.stop()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.error_catching[("word", self.job_name)] = "\n".join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)
            )


class PhoneCtmProcessWorker(mp.Process):
    """
    Multiprocessing worker for loading phone CTM files

    See Also
    --------
    :meth:`.CorpusAligner.ctms_to_textgrids_mp`
        Main function that runs this worker in parallel

    Parameters
    ----------
    job_name: int
        Job name
    to_process_queue: :class:`~multiprocessing.Queue`
        Return queue of jobs for later workers to process
    stopped: :class:`~montreal_forced_aligner.utils.Stopped`
        Stop check for processing
    error_catching: dict[tuple[str, int], str]
        Dictionary for storing errors encountered
    arguments: :class:`~montreal_forced_aligner.alignment.multiprocessing.PhoneCtmArguments`
        Arguments to pass to the CTM processing function
    """

    def __init__(
        self,
        job_name: int,
        to_process_queue: mp.Queue,
        stopped: Stopped,
        error_catching: CtmErrorDict,
        arguments: PhoneCtmArguments,
    ):
        mp.Process.__init__(self)
        self.job_name = job_name
        self.dictionaries = arguments.dictionaries
        self.ctm_paths = arguments.ctm_paths
        self.to_process_queue = to_process_queue
        self.stopped = stopped
        self.error_catching = error_catching

        self.utterances = arguments.utterances

        self.reversed_phone_mappings = arguments.reversed_phone_mappings
        self.positions = arguments.positions

    def run(self) -> None:
        """Run the phone processing"""
        cur_utt = None
        cur_file = None
        utt_begin = 0
        current_labels = []

        current_file_data = {}

        def process_current_utt(cur_utt: Utterance, current_labels: list[CtmInterval]) -> None:
            """Process current stack of intervals"""
            actual_labels = parse_from_phone(
                current_labels, self.reversed_phone_mappings[dict_name], self.positions[dict_name]
            )
            current_file_data[cur_utt.name] = actual_labels

        def process_current_file(cur_file: str) -> None:
            """Process current file and add to return queue"""
            self.to_process_queue.put(("phone", cur_file, current_file_data))

        try:
            for dict_name in self.dictionaries:
                with open(self.ctm_paths[dict_name], "r") as word_file:
                    for line in word_file:
                        line = line.strip()
                        if not line:
                            continue
                        interval = process_ctm_line(line)
                        utt = interval.utterance
                        if cur_utt is None:
                            cur_utt = self.utterances[dict_name][utt]
                            cur_file = cur_utt.file_name
                            utt_begin = cur_utt.begin

                        if utt != cur_utt:

                            process_current_utt(cur_utt, current_labels)

                            cur_utt = self.utterances[dict_name][utt]
                            file_name = cur_utt.file_name
                            utt_begin = cur_utt.begin

                            if file_name != cur_file:
                                process_current_file(cur_file)
                                current_file_data = {}
                                cur_file = file_name
                            current_labels = []
                        if utt_begin:
                            interval.shift_times(utt_begin)
                        current_labels.append(interval)
                if current_labels:
                    process_current_utt(cur_utt, current_labels)
                    process_current_file(cur_file)
        except Exception:
            self.stopped.stop()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.error_catching[("phone", self.job_name)] = (
                "\n".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                + f"\n\n{len(self.utterances['english'].keys())}\nCould not find: {utt}\n"
                + "\n".join(self.utterances["english"].keys())
            )


class CombineProcessWorker(mp.Process):
    """
    Multiprocessing worker for loading phone CTM files

    See Also
    --------
    :meth:`.CorpusAligner.ctms_to_textgrids_mp`
        Main function that runs this worker in parallel

    Parameters
    ----------
    job_name: int
        Job name
    to_process_queue: :class:`~multiprocessing.Queue`
        Input queue of phone and word ctms to combine
    to_export_queue: :class:`~multiprocessing.Queue`
        Export queue of combined CTMs
    stopped: :class:`~montreal_forced_aligner.utils.Stopped`
        Stop check for processing
    finished_combining: :class:`~montreal_forced_aligner.utils.Stopped`
        Signal that this worker has finished combining all CTMs
    error_catching: dict[tuple[str, int], str]
        Dictionary for storing errors encountered
    arguments: :class:`~montreal_forced_aligner.alignment.multiprocessing.CombineCtmArguments`
        Arguments to pass to the CTM combining function
    """

    def __init__(
        self,
        job_name: int,
        to_process_queue: mp.Queue,
        to_export_queue: mp.Queue,
        stopped: Stopped,
        finished_combining: Stopped,
        error_catching: CtmErrorDict,
        arguments: CombineCtmArguments,
    ):
        mp.Process.__init__(self)
        self.job_name = job_name
        self.to_process_queue = to_process_queue
        self.to_export_queue = to_export_queue
        self.stopped = stopped
        self.finished_combining = finished_combining
        self.error_catching = error_catching

        self.files = arguments.files
        self.speakers = arguments.speakers
        self.dictionary_data = arguments.dictionary_data
        self.cleanup_textgrids = arguments.cleanup_textgrids

        for file in self.files.values():
            for s in file.speaker_ordering:
                if s.name not in self.speakers:
                    continue
                s.dictionary_data = self.dictionary_data[self.speakers[s.name].dictionary_name]

    def run(self) -> None:
        """Run the combination function"""

        phone_data = {}
        word_data = {}
        while True:
            try:
                w_p, file_name, data = self.to_process_queue.get(timeout=queue_polling_timeout)
            except Empty:
                if self.finished_combining.stop_check():
                    break
                continue
            self.to_process_queue.task_done()
            if self.stopped.stop_check():
                continue
            if w_p == "phone":
                if file_name in word_data:
                    word_ctm = word_data.pop(file_name)
                    phone_ctm = data
                else:
                    phone_data[file_name] = data
                    continue
            else:
                if file_name in phone_data:
                    phone_ctm = phone_data.pop(file_name)
                    word_ctm = data
                else:
                    word_data[file_name] = data
                    continue
            try:
                file = self.files[file_name]
                for u_name, u in file.utterances.items():
                    if u_name not in word_ctm:
                        continue
                    u.speaker.dictionary_data = self.dictionary_data[
                        self.speakers[u.speaker_name].dictionary_name
                    ]
                    u.word_labels = word_ctm[u_name]
                    u.phone_labels = phone_ctm[u_name]
                processed_check = True
                for s in file.speaker_ordering:
                    if s.name not in self.speakers:
                        continue
                    if not file.has_fully_aligned_speaker(s):
                        processed_check = False
                        break
                if not processed_check:
                    continue
                data = generate_tiers(file, cleanup_textgrids=self.cleanup_textgrids)
                self.to_export_queue.put((file_name, data))
            except Exception:
                self.stopped.stop()
                exc_type, exc_value, exc_traceback = sys.exc_info()
                self.error_catching[("combining", self.job_name)] = "\n".join(
                    traceback.format_exception(exc_type, exc_value, exc_traceback)
                )


class ExportTextGridProcessWorker(mp.Process):
    """
    Multiprocessing worker for exporting TextGrids

    See Also
    --------
    :meth:`.CorpusAligner.ctms_to_textgrids_mp`
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
    """

    def __init__(
        self,
        for_write_queue: mp.Queue,
        stopped: Stopped,
        finished_processing: Stopped,
        textgrid_errors: dict[str, str],
        arguments: ExportTextGridArguments,
    ):
        mp.Process.__init__(self)
        self.for_write_queue = for_write_queue
        self.stopped = stopped
        self.finished_processing = finished_processing
        self.textgrid_errors = textgrid_errors

        self.files = arguments.files
        self.output_directory = arguments.output_directory
        self.backup_output_directory = arguments.backup_output_directory

        self.frame_shift = arguments.frame_shift

    def run(self) -> None:
        """Run the exporter function"""
        while True:
            try:
                file_name, data = self.for_write_queue.get(timeout=queue_polling_timeout)
            except Empty:
                if self.finished_processing.stop_check():
                    break
                continue
            self.for_write_queue.task_done()
            if self.stopped.stop_check():
                continue
            try:
                overwrite = True
                file = self.files[file_name]
                output_path = file.construct_output_path(
                    self.output_directory, self.backup_output_directory
                )

                export_textgrid(file, output_path, data, self.frame_shift, overwrite)
            except Exception:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                self.textgrid_errors[file_name] = "\n".join(
                    traceback.format_exception(exc_type, exc_value, exc_traceback)
                )


class ExportPreparationProcessWorker(mp.Process):
    """
    Multiprocessing worker for preparing CTMs for export

    See Also
    --------
    :meth:`.CorpusAligner.ctms_to_textgrids_mp`
        Main function that runs this worker in parallel

    Parameters
    ----------
    to_export_queue: :class:`~multiprocessing.Queue`
        Input queue of combined CTMs
    for_write_queue: :class:`~multiprocessing.Queue`
        Export queue of files to export
    stopped: :class:`~montreal_forced_aligner.utils.Stopped`
        Stop check for processing
    finished_combining: :class:`~montreal_forced_aligner.utils.Stopped`
        Input signal that all CTMs have been combined
    files: dict[str, File]
        Files in corpus
    """

    def __init__(
        self,
        to_export_queue: mp.Queue,
        for_write_queue: mp.Queue,
        stopped: Stopped,
        finished_combining: Stopped,
        files: dict[str, File],
    ):
        mp.Process.__init__(self)
        self.to_export_queue = to_export_queue
        self.for_write_queue = for_write_queue
        self.stopped = stopped
        self.finished_combining = finished_combining

        self.files = files

    def run(self) -> None:
        """Run the export preparation worker"""
        export_data = {}
        try:
            while True:
                try:
                    file_name, data = self.to_export_queue.get(timeout=queue_polling_timeout)
                except Empty:
                    if self.finished_combining.stop_check():
                        break
                    continue
                self.to_export_queue.task_done()
                if self.stopped.stop_check():
                    continue
                file = self.files[file_name]
                if len(file.speaker_ordering) > 1:
                    if file_name not in export_data:
                        export_data[file_name] = data
                    else:
                        export_data[file_name].update(data)
                    if len(export_data[file_name]) == len(file.speaker_ordering):
                        data = export_data.pop(file_name)
                        self.for_write_queue.put((file_name, data))
                else:
                    self.for_write_queue.put((file_name, data))

            for k, v in export_data.items():
                self.for_write_queue.put((k, v))
        except Exception:
            self.stopped.stop()
            raise
