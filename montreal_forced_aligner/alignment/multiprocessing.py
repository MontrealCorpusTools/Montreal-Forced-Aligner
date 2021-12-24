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
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Union

from montreal_forced_aligner.textgrid import export_textgrid, process_ctm_line
from montreal_forced_aligner.utils import Stopped, thirdparty_binary

if TYPE_CHECKING:
    from montreal_forced_aligner.abc import CtmErrorDict, MetaDict

__all__ = [
    "WordCtmProcessWorker",
    "PhoneCtmProcessWorker",
    "ExportTextGridProcessWorker",
    "WordCtmArguments",
    "PhoneCtmArguments",
    "ExportTextGridArguments",
    "align_func",
    "AlignArguments",
    "ali_to_ctm_func",
    "AliToCtmArguments",
    "compile_information_func",
    "CompileInformationArguments",
    "compile_train_graphs_func",
    "CompileTrainGraphsArguments",
]


class AliToCtmArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.alignment.multiprocessing.ali_to_ctm_func`"""

    log_path: str
    dictionaries: List[str]
    ali_paths: Dict[str, str]
    text_int_paths: Dict[str, str]
    word_boundary_int_paths: Dict[str, str]
    frame_shift: float
    model_path: str
    ctm_paths: Dict[str, str]
    word_mode: bool


class WordCtmArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.NoCleanupWordCtmProcessWorker`"""

    ctm_paths: Dict[str, str]
    dictionaries: List[str]


class PhoneCtmArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.PhoneCtmProcessWorker`"""

    ctm_paths: Dict[str, str]
    dictionaries: List[str]


class ExportTextGridArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.ExportTextGridProcessWorker`"""

    log_path: str
    frame_shift: int
    output_directory: str
    backup_output_directory: str


class CompileInformationArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.alignment.multiprocessing.compile_information_func`"""

    align_log_paths: str


class CompileTrainGraphsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.alignment.multiprocessing.compile_train_graphs_func`"""

    log_path: str
    dictionaries: List[str]
    tree_path: str
    model_path: str
    text_int_paths: Dict[str, str]
    disambig_paths: Dict[str, str]
    lexicon_fst_paths: Dict[str, str]
    fst_scp_paths: Dict[str, str]


class AlignArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.alignment.multiprocessing.align_func`"""

    log_path: str
    dictionaries: List[str]
    fst_scp_paths: Dict[str, str]
    feature_strings: Dict[str, str]
    model_path: str
    ali_paths: Dict[str, str]
    align_options: MetaDict


def compile_train_graphs_func(
    log_path: str,
    dictionaries: List[str],
    tree_path: str,
    model_path: str,
    text_int_paths: Dict[str, str],
    disambig_path: str,
    lexicon_fst_paths: Dict[str, str],
    fst_scp_paths: Dict[str, str],
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
    dictionaries: List[str],
    fst_scp_paths: Dict[str, str],
    feature_strings: Dict[str, str],
    model_path: str,
    ali_paths: Dict[str, str],
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


def compile_information_func(align_log_path: str) -> Dict[str, Union[List[str], float, int]]:
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
    dictionaries: List[str],
    ali_paths: Dict[str, str],
    text_int_paths: Dict[str, str],
    word_boundary_int_paths: Dict[str, str],
    frame_shift: float,
    model_path: str,
    ctm_paths: Dict[str, str],
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


class WordCtmProcessWorker(mp.Process):
    """
    Multiprocessing worker for loading word CTM files

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
    arguments: :class:`~montreal_forced_aligner.alignment.multiprocessing.WordCtmArguments`
        Arguments to pass to the CTM processing function
    """

    def __init__(
        self,
        job_name: int,
        to_process_queue: mp.Queue,
        stopped: Stopped,
        error_catching: CtmErrorDict,
        arguments: WordCtmArguments,
        finished_signal: Stopped,
    ):
        mp.Process.__init__(self)
        self.job_name = job_name
        self.dictionaries = arguments.dictionaries
        self.ctm_paths = arguments.ctm_paths
        self.to_process_queue = to_process_queue
        self.stopped = stopped
        self.error_catching = error_catching
        self.finished_signal = finished_signal

        self.arguments = arguments

    def run(self) -> None:
        """
        Run the word processing with no clean up
        """
        cur_utt = None
        intervals = []
        try:
            for dict_name in self.dictionaries:
                ctm_path = self.ctm_paths[dict_name]
                with open(ctm_path, "r") as word_file:
                    for line in word_file:
                        line = line.strip()
                        if not line:
                            continue
                        interval = process_ctm_line(line)
                        if cur_utt is None:
                            cur_utt = interval.utterance
                        if cur_utt != interval.utterance:

                            self.to_process_queue.put(("word", intervals))
                            intervals = []
                            cur_utt = interval.utterance
                        intervals.append(interval)

        except Exception:
            self.stopped.stop()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.error_catching[("word", self.job_name)] = "\n".join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)
            )
        finally:
            self.finished_signal.stop()


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
        finished_signal: Stopped,
    ):
        mp.Process.__init__(self)
        self.job_name = job_name
        self.dictionaries = arguments.dictionaries
        self.ctm_paths = arguments.ctm_paths
        self.to_process_queue = to_process_queue
        self.stopped = stopped
        self.error_catching = error_catching
        self.finished_signal = finished_signal

    def run(self) -> None:
        """Run the phone processing"""
        cur_utt = None
        intervals = []
        try:
            for dict_name in self.dictionaries:
                ctm_path = self.ctm_paths[dict_name]
                with open(ctm_path, "r") as word_file:
                    for line in word_file:
                        line = line.strip()
                        if not line:
                            continue
                        interval = process_ctm_line(line)
                        if cur_utt is None:
                            cur_utt = interval.utterance
                        if cur_utt != interval.utterance:

                            self.to_process_queue.put(("phone", intervals))
                            intervals = []
                            cur_utt = interval.utterance
                        intervals.append(interval)

        except Exception:
            self.stopped.stop()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.error_catching[("phone", self.job_name)] = traceback.format_exception(
                exc_type, exc_value, exc_traceback
            )
        finally:
            self.finished_signal.stop()


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
        textgrid_errors: Dict[str, str],
        arguments: ExportTextGridArguments,
    ):
        mp.Process.__init__(self)
        self.for_write_queue = for_write_queue
        self.stopped = stopped
        self.finished_processing = finished_processing
        self.textgrid_errors = textgrid_errors

        self.output_directory = arguments.output_directory
        self.backup_output_directory = arguments.backup_output_directory

        self.frame_shift = arguments.frame_shift

    def run(self) -> None:
        """Run the exporter function"""
        while True:
            try:
                data, output_path, duration = self.for_write_queue.get(timeout=1)
            except Empty:
                if self.finished_processing.stop_check():
                    break
                continue
            self.for_write_queue.task_done()
            if self.stopped.stop_check():
                continue
            try:
                export_textgrid(data, output_path, duration, self.frame_shift)
            except Exception:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                self.textgrid_errors[output_path] = "\n".join(
                    traceback.format_exception(exc_type, exc_value, exc_traceback)
                )
