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

from montreal_forced_aligner.dictionary.multispeaker import MultispeakerSanitizationFunction
from montreal_forced_aligner.textgrid import export_textgrid, process_ctm_line
from montreal_forced_aligner.utils import KaldiFunction, Stopped, thirdparty_binary

if TYPE_CHECKING:
    from montreal_forced_aligner.abc import MetaDict
    from montreal_forced_aligner.data import CtmInterval

__all__ = [
    "WordAlignmentFunction",
    "PhoneAlignmentFunction",
    "ExportTextGridProcessWorker",
    "WordAlignmentArguments",
    "PhoneAlignmentArguments",
    "ExportTextGridArguments",
    "AlignFunction",
    "AlignArguments",
    "compile_information_func",
    "CompileInformationArguments",
    "CompileTrainGraphsFunction",
    "CompileTrainGraphsArguments",
]


class WordAlignmentArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.WordAlignmentFunction`"""

    log_path: str
    model_path: str
    frame_shift: float
    cleanup_textgrids: bool
    oov_word: str
    sanitize_function: MultispeakerSanitizationFunction
    dictionaries: List[str]
    ali_paths: Dict[str, str]
    word_boundary_int_paths: Dict[str, str]
    text_int_paths: Dict[str, str]
    reversed_word_mappings: Dict[str, Dict[int, str]]
    utterance_texts: Dict[str, Dict[str, str]]
    utterance_speakers: Dict[str, Dict[str, str]]


class PhoneAlignmentArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.PhoneAlignmentFunction`"""

    log_path: str
    model_path: str
    frame_shift: float
    position_dependent_phones: bool
    cleanup_textgrids: bool
    silence_phone: str
    dictionaries: List[str]
    ali_paths: Dict[str, str]
    word_boundary_int_paths: Dict[str, str]
    text_int_paths: Dict[str, str]
    reversed_phone_mapping: Dict[int, str]


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
    """Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.CompileTrainGraphsFunction`"""

    log_path: str
    dictionaries: List[str]
    tree_path: str
    model_path: str
    text_int_paths: Dict[str, str]
    disambig_path: str
    lexicon_fst_paths: Dict[str, str]
    fst_scp_paths: Dict[str, str]


class AlignArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.AlignFunction`"""

    log_path: str
    dictionaries: List[str]
    fst_scp_paths: Dict[str, str]
    feature_strings: Dict[str, str]
    model_path: str
    ali_paths: Dict[str, str]
    align_options: MetaDict


class AccStatsArguments(NamedTuple):
    """
    Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.AccStatsFunction`
    """

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ali_paths: Dict[str, str]
    acc_paths: Dict[str, str]
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
        self.log_path = args.log_path
        self.dictionaries = args.dictionaries
        self.feature_strings = args.feature_strings
        self.model_path = args.model_path
        self.ali_paths = args.ali_paths
        self.acc_paths = args.acc_paths

    def run(self):
        """Run the function"""
        processed_count = 0
        with open(self.log_path, "w", encoding="utf8") as log_file:
            for dict_name in self.dictionaries:
                acc_proc = subprocess.Popen(
                    [
                        thirdparty_binary("gmm-acc-stats-ali"),
                        self.model_path,
                        self.feature_strings[dict_name],
                        f"ark,s,cs:{self.ali_paths[dict_name]}",
                        self.acc_paths[dict_name],
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
                    yield utterance, float(log_likelihood)


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


class WordAlignmentFunction(KaldiFunction):

    """
    Multiprocessing function to collect word alignments from the aligned lattice

    See Also
    --------
    :meth:`.CorpusAligner.collect_word_alignments`
        Main function that calls this function in parallel
    :meth:`.CorpusAligner.word_alignments_arguments`
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
    arguments: :class:`~montreal_forced_aligner.alignment.multiprocessing.WordAlignmentArguments`
        Arguments for the function
    """

    def __init__(self, arguments: WordAlignmentArguments):
        self.arguments = arguments

    def cleanup_intervals(self, utterance_name: str, intervals: List[CtmInterval]):
        from montreal_forced_aligner.data import CtmInterval

        speaker = None
        for utt2spk in self.arguments.utterance_speakers.values():
            if utterance_name in utt2spk:
                speaker = utt2spk[utterance_name]
                break
        dict_name = self.arguments.sanitize_function.get_dict_name_for_speaker(speaker)
        mapping = self.arguments.reversed_word_mappings[dict_name]
        for interval in intervals:
            interval.label = mapping[int(interval.label)]
        if not self.arguments.cleanup_textgrids:
            return intervals

        text = self.arguments.utterance_texts[dict_name][utterance_name]
        sanitize, split = self.arguments.sanitize_function.get_functions_for_speaker(speaker)
        if split is None:
            return intervals
        cur_ind = 0
        try:
            actual_labels = []
            for word in text.split():
                splits = split(word)
                b = 1000000
                e = -1
                for w in splits:
                    if not w:
                        continue
                    cur = intervals[cur_ind]
                    if w == cur.label or cur.label == self.arguments.oov_word:
                        if cur.begin < b:
                            b = cur.begin
                        if cur.end > e:
                            e = cur.end
                    cur_ind += 1
                lab = CtmInterval(b, e, word, utterance_name)
                actual_labels.append(lab)
        except Exception:
            print("Error parsing:")
            print(text)
            for word in text.split():
                print(word)
                splits = split(word)
                for w in splits:
                    print(w)

            raise
        return actual_labels

    def run(self):
        """Run the function"""
        with open(self.arguments.log_path, "w", encoding="utf8") as log_file:
            for dict_name in self.arguments.dictionaries:
                cur_utt = None
                intervals = []
                ali_path = self.arguments.ali_paths[dict_name]
                text_int_path = self.arguments.text_int_paths[dict_name]
                word_boundary_int_path = self.arguments.word_boundary_int_paths[dict_name]
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
                        self.arguments.model_path,
                        "ark:-",
                        "ark:-",
                    ],
                    stdin=lin_proc.stdout,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=os.environ,
                )
                nbest_proc = subprocess.Popen(
                    [
                        thirdparty_binary("nbest-to-ctm"),
                        "--print-args=false",
                        f"--frame-shift={self.arguments.frame_shift}",
                        "ark:-",
                        "-",
                    ],
                    stderr=log_file,
                    stdin=align_words_proc.stdout,
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
                        yield cur_utt, self.cleanup_intervals(cur_utt, intervals)
                        intervals = []
                        cur_utt = interval.utterance
                    intervals.append(interval)
            if intervals:
                yield cur_utt, self.cleanup_intervals(cur_utt, intervals)


class PhoneAlignmentFunction(KaldiFunction):

    """
    Multiprocessing function to collect phone alignments from the aligned lattice

    See Also
    --------
    :meth:`.CorpusAligner.collect_phone_alignments`
        Main function that calls this function in parallel
    :meth:`.CorpusAligner.phone_alignments_arguments`
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
    arguments: :class:`~montreal_forced_aligner.alignment.multiprocessing.PhoneAlignmentArguments`
        Arguments for the function
    """

    def __init__(self, arguments: PhoneAlignmentArguments):
        self.arguments = arguments

    def cleanup_intervals(self, intervals: List[CtmInterval]):
        actual_labels = []
        for interval in intervals:
            label = self.arguments.reversed_phone_mapping[int(interval.label)]
            if self.arguments.position_dependent_phones and "_" in label:
                label = label[:-2]
            interval.label = label
            if self.arguments.cleanup_textgrids and interval.label == self.arguments.silence_phone:
                continue
            actual_labels.append(interval)
        return actual_labels

    def run(self):
        """Run the function"""
        with open(self.arguments.log_path, "w", encoding="utf8") as log_file:
            for dict_name in self.arguments.dictionaries:
                cur_utt = None
                intervals = []
                ali_path = self.arguments.ali_paths[dict_name]
                text_int_path = self.arguments.text_int_paths[dict_name]
                word_boundary_int_path = self.arguments.word_boundary_int_paths[dict_name]
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
                        self.arguments.model_path,
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
                        self.arguments.model_path,
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
                        f"--frame-shift={self.arguments.frame_shift}",
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

                        yield cur_utt, self.cleanup_intervals(intervals)
                        intervals = []
                        cur_utt = interval.utterance
                    intervals.append(interval)
            if intervals:
                yield cur_utt, self.cleanup_intervals(intervals)


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
                    log_file.write(f"  * {len(data)} speakers\n")
                    for speaker, d in data.items():

                        log_file.write(f"Speaker {speaker}:\n")
                        log_file.write(f"  * {len(d['words'])} words\n")
                        log_file.write(f"  * {len(d['phones'])} phones\n")
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
