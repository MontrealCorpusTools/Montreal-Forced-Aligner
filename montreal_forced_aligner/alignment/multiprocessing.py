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
from montreal_forced_aligner.utils import KaldiFunction, Stopped, thirdparty_binary

if TYPE_CHECKING:
    from montreal_forced_aligner.abc import CtmErrorDict, MetaDict

__all__ = [
    "WordCtmProcessWorker",
    "PhoneCtmProcessWorker",
    "ExportTextGridProcessWorker",
    "WordCtmArguments",
    "PhoneCtmArguments",
    "ExportTextGridArguments",
    "AlignFunction",
    "AlignArguments",
    "AliToCtmFunction",
    "AliToCtmArguments",
    "compile_information_func",
    "CompileInformationArguments",
    "CompileTrainGraphsFunction",
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
    """Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.WordCtmProcessWorker`"""

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
    disambig_path: str
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
        self.log_path = args.log_path
        self.dictionaries = args.dictionaries
        self.tree_path = args.tree_path
        self.model_path = args.model_path
        self.text_int_paths = args.text_int_paths
        self.disambig_path = args.disambig_path
        self.lexicon_fst_paths = args.lexicon_fst_paths
        self.fst_scp_paths = args.fst_scp_paths

    def run(self):
        with open(self.log_path, "w", encoding="utf8") as log_file:
            for dict_name in self.dictionaries:
                fst_scp_path = self.fst_scp_paths[dict_name]
                fst_ark_path = fst_scp_path.replace(".scp", ".ark")
                text_path = self.text_int_paths[dict_name]
                proc = subprocess.Popen(
                    [
                        thirdparty_binary("compile-train-graphs"),
                        f"--read-disambig-syms={self.disambig_path}",
                        self.tree_path,
                        self.model_path,
                        self.lexicon_fst_paths[dict_name],
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

    progress_pattern = re.compile(r"^LOG.*(?P<utterance>.*)")

    def __init__(self, args: AlignArguments):
        self.log_path = args.log_path
        self.dictionaries = args.dictionaries
        self.fst_scp_paths = args.fst_scp_paths
        self.feature_strings = args.feature_strings
        self.model_path = args.model_path
        self.ali_paths = args.ali_paths
        self.align_options = args.align_options

    def run(self):
        """Run the function"""
        with open(self.log_path, "w", encoding="utf8") as log_file:
            for dict_name in self.dictionaries:
                feature_string = self.feature_strings[dict_name]
                fst_path = self.fst_scp_paths[dict_name]
                ali_path = self.ali_paths[dict_name]
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
                    stderr=subprocess.PIPE,
                    encoding="utf8",
                    stdin=boost_proc.stdout,
                    env=os.environ,
                )
                for line in align_proc.stderr:
                    log_file.write(line)
                    m = self.progress_pattern.match(line.strip())
                    if m:
                        yield m.group("utterance")


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


class AliToCtmFunction(KaldiFunction):

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
    args: :class:`~montreal_forced_aligner.alignment.multiprocessing.AliToCtmArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(
        r"^LOG.* Converted (?P<done>\d+) linear lattices to ctm format; (?P<errors>\d+) had errors."
    )

    def __init__(self, args: AliToCtmArguments):
        self.log_path = args.log_path
        self.dictionaries = args.dictionaries
        self.ali_paths = args.ali_paths
        self.text_int_paths = args.text_int_paths
        self.word_boundary_int_paths = args.word_boundary_int_paths
        self.frame_shift = args.frame_shift
        self.model_path = args.model_path
        self.ctm_paths = args.ctm_paths
        self.word_mode = args.word_mode

    def run(self):
        """Run the function"""
        with open(self.log_path, "w", encoding="utf8") as log_file:
            for dict_name in self.dictionaries:
                ali_path = self.ali_paths[dict_name]
                text_int_path = self.text_int_paths[dict_name]
                ctm_path = self.ctm_paths[dict_name]
                word_boundary_int_path = self.word_boundary_int_paths[dict_name]
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
                        self.model_path,
                        "ark:-",
                        "ark:-",
                    ],
                    stdin=lin_proc.stdout,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=os.environ,
                )
                if self.word_mode:
                    nbest_proc = subprocess.Popen(
                        [
                            thirdparty_binary("nbest-to-ctm"),
                            f"--frame-shift={self.frame_shift}",
                            "ark:-",
                            ctm_path,
                        ],
                        stderr=subprocess.PIPE,
                        stdin=align_words_proc.stdout,
                        env=os.environ,
                        encoding="utf8",
                    )
                else:
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
                            f"--frame-shift={self.frame_shift}",
                            "ark:-",
                            ctm_path,
                        ],
                        stdin=phone_proc.stdout,
                        stderr=subprocess.PIPE,
                        env=os.environ,
                        encoding="utf8",
                    )
                for line in nbest_proc.stderr:
                    m = self.progress_pattern.match(line.strip())
                    if m:
                        yield int(m.group("done")), int(m.group("errors"))


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
    ):
        mp.Process.__init__(self)
        self.job_name = job_name
        self.dictionaries = arguments.dictionaries
        self.ctm_paths = arguments.ctm_paths
        self.to_process_queue = to_process_queue
        self.stopped = stopped
        self.error_catching = error_catching
        self.finished_signal = Stopped()

        self.arguments = arguments

    def run(self) -> None:
        """
        Run the word processing
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
                if intervals:
                    self.to_process_queue.put(("word", intervals))

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
    ):
        mp.Process.__init__(self)
        self.job_name = job_name
        self.dictionaries = arguments.dictionaries
        self.ctm_paths = arguments.ctm_paths
        self.to_process_queue = to_process_queue
        self.stopped = stopped
        self.error_catching = error_catching
        self.finished_signal = Stopped()

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
                if intervals:
                    self.to_process_queue.put(("phone", intervals))

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
        self.log_path = arguments.log_path

    def run(self) -> None:
        """Run the exporter function"""
        with open(self.log_path, "w", encoding="utf8") as log_file:
            while True:
                try:
                    data, output_path, duration = self.for_write_queue.get(timeout=1)
                    log_file.write(f"Processing {output_path}...\n")
                except Empty:
                    if self.finished_processing.stop_check():
                        break
                    continue
                self.for_write_queue.task_done()
                if self.stopped.stop_check():
                    continue
                try:
                    export_textgrid(data, output_path, duration, self.frame_shift)
                    log_file.write("Done!\n")
                except Exception:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    self.textgrid_errors[output_path] = "\n".join(
                        traceback.format_exception(exc_type, exc_value, exc_traceback)
                    )
