"""
Utility functions
=================

"""
from __future__ import annotations

import datetime
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import time
import typing
from queue import Empty
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import sqlalchemy
from sqlalchemy.orm import Session

from montreal_forced_aligner.abc import KaldiFunction
from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.data import CtmInterval, DatasetType, MfaArguments
from montreal_forced_aligner.db import Corpus, Dictionary
from montreal_forced_aligner.exceptions import (
    DictionaryError,
    KaldiProcessingError,
    MultiprocessingError,
    ThirdpartyError,
)
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.textgrid import process_ctm_line

__all__ = [
    "thirdparty_binary",
    "log_kaldi_errors",
    "get_mfa_version",
    "parse_logs",
    "Counter",
    "Stopped",
    "ProcessWorker",
    "KaldiProcessWorker",
    "run_mp",
    "run_non_mp",
    "run_kaldi_function",
]
canary_kaldi_bins = [
    "compute-mfcc-feats",
    "compute-and-process-kaldi-pitch-feats",
    "gmm-align-compiled",
    "gmm-est-fmllr",
    "gmm-est-fmllr-gpost",
    "lattice-oracle",
    "gmm-latgen-faster",
    "fstdeterminizestar",
    "fsttablecompose",
    "gmm-rescore-lattice",
]

logger = logging.getLogger("mfa")


def inspect_database(name: str) -> DatasetType:
    """
    Inspect the database file to generate its DatasetType

    Parameters
    ----------
    name: str
        Name of database

    Returns
    -------
    DatasetType
        Dataset type of the database
    """

    string = (
        f"postgresql+psycopg2://localhost:{GLOBAL_CONFIG.current_profile.database_port}/{name}"
    )
    try:
        engine = sqlalchemy.create_engine(
            string,
            poolclass=sqlalchemy.NullPool,
            pool_reset_on_return=None,
            isolation_level="AUTOCOMMIT",
            logging_name="inspect_dataset_engine",
        ).execution_options(logging_token="inspect_dataset_engine")
        with Session(engine) as session:
            corpus = session.query(Corpus).first()
            dictionary = session.query(Dictionary).first()
            if corpus is None and dictionary is None:
                return DatasetType.NONE
            elif corpus is None:
                return DatasetType.DICTIONARY
            elif dictionary is None:
                if corpus.has_sound_files:
                    return DatasetType.ACOUSTIC_CORPUS
                else:
                    return DatasetType.TEXT_CORPUS
            if corpus.has_sound_files:
                return DatasetType.ACOUSTIC_CORPUS_WITH_DICTIONARY
            else:
                return DatasetType.TEXT_CORPUS_WITH_DICTIONARY
    except (sqlalchemy.exc.OperationalError, sqlalchemy.exc.ProgrammingError):
        return DatasetType.NONE


def get_class_for_dataset_type(dataset_type: DatasetType):
    """
    Generate the corresponding MFA class for a given DatasetType

    Parameters
    ----------
    dataset_type: DatasetType
        Dataset type for the class

    Returns
    -------
    Union[None, AcousticCorpus, TextCorpus, AcousticCorpusWithPronunciations, DictionaryTextCorpus,MultispeakerDictionary]
        Class to use for the current database file
    """
    from montreal_forced_aligner.corpus.acoustic_corpus import (
        AcousticCorpus,
        AcousticCorpusWithPronunciations,
    )
    from montreal_forced_aligner.corpus.text_corpus import DictionaryTextCorpus, TextCorpus
    from montreal_forced_aligner.dictionary import MultispeakerDictionary

    mapping = {
        DatasetType.NONE: None,
        DatasetType.ACOUSTIC_CORPUS: AcousticCorpus,
        DatasetType.TEXT_CORPUS: TextCorpus,
        DatasetType.ACOUSTIC_CORPUS_WITH_DICTIONARY: AcousticCorpusWithPronunciations,
        DatasetType.TEXT_CORPUS_WITH_DICTIONARY: DictionaryTextCorpus,
        DatasetType.DICTIONARY: MultispeakerDictionary,
    }
    return mapping[dataset_type]


def parse_dictionary_file(
    path: str,
) -> typing.Generator[
    typing.Tuple[
        str,
        typing.List[str],
        typing.Optional[float],
        typing.Optional[float],
        typing.Optional[float],
        typing.Optional[float],
    ]
]:
    """
    Parses a lexicon file and yields parsed pronunciation lines

    Parameters
    ----------
    path: str
        Path to lexicon file

    Yields
    ------
    str
        Orthographic word
    list[str]
        Pronunciation
    float or None
        Pronunciation probability
    float or None
        Probability of silence following the pronunciation
    float or None
        Correction factor for silence before the pronunciation
    float or None
        Correction factor for no silence before the pronunciation
    """
    prob_pattern = re.compile(r"\b\d+\.\d+\b")
    with mfa_open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            line = line.split()
            if len(line) <= 1:
                raise DictionaryError(
                    f'Error parsing line {i} of {path}: "{line}" did not have a pronunciation'
                )
            word = line.pop(0)
            prob = None
            silence_after_prob = None
            silence_before_correct = None
            non_silence_before_correct = None
            if prob_pattern.match(line[0]):
                prob = float(line.pop(0))
                if prob_pattern.match(line[0]):
                    silence_after_prob = float(line.pop(0))
                    if prob_pattern.match(line[0]):
                        silence_before_correct = float(line.pop(0))
                        if prob_pattern.match(line[0]):
                            non_silence_before_correct = float(line.pop(0))
            pron = tuple(line)
            yield word, pron, prob, silence_after_prob, silence_before_correct, non_silence_before_correct


def parse_ctm_output(
    proc: subprocess.Popen, reversed_phone_mapping: Dict[int, Any], raw_id: bool = False
) -> typing.Generator[typing.Tuple[typing.Union[int, str], typing.List[CtmInterval]]]:
    """
    Parse stdout of a process into intervals grouped by utterance

    Parameters
    ----------
    proc: :class:`subprocess.Popen`
    reversed_phone_mapping: dict[int, Any]
        Mapping from kaldi integer IDs to phones
    raw_id: bool
        Flag for returning the kaldi internal ID of the utterance rather than its integer ID

    Yields
    -------
    int or str
        Utterance ID
    list[:class:`~montreal_forced_aligner.data.CtmInterval`]
        List of CTM intervals for the utterance
    """
    current_utt = None
    intervals = []
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            utt, interval = process_ctm_line(line, reversed_phone_mapping, raw_id=raw_id)
        except ValueError:
            continue
        if current_utt is None:
            current_utt = utt
        if current_utt != utt:
            yield current_utt, intervals
            intervals = []
            current_utt = utt
        intervals.append(interval)
    if intervals:
        yield current_utt, intervals


def get_mfa_version() -> str:
    """
    Get the current MFA version

    Returns
    -------
    str
        MFA version
    """
    try:
        from ._version import version as __version__  # noqa
    except ImportError:
        __version__ = "2.0.0"
    return __version__


def check_third_party():
    """
    Checks whether third party software is available on the path

    Raises
    -------
    :class:`~montreal_forced_aligner.exceptions.ThirdpartyError`
    """
    bin_path = shutil.which("sox")
    if bin_path is None:
        raise ThirdpartyError("sox")
    bin_path = shutil.which("fstcompile")
    if bin_path is None:
        raise ThirdpartyError("fstcompile", open_fst=True)

    p = subprocess.run(["fstcompile", "--help"], capture_output=True, text=True)
    if p.returncode == 1 and p.stderr:
        raise ThirdpartyError("fstcompile", open_fst=True, error_text=p.stderr)
    for fn in canary_kaldi_bins:
        try:
            p = subprocess.run([thirdparty_binary(fn), "--help"], capture_output=True, text=True)
        except Exception as e:
            raise ThirdpartyError(fn, error_text=str(e))
        if p.returncode == 1 and p.stderr:
            raise ThirdpartyError(fn, error_text=p.stderr)


def thirdparty_binary(binary_name: str) -> str:
    """
    Generate full path to a given binary name

    Notes
    -----
    With the move to conda, this function is deprecated as conda will manage the path much better

    Parameters
    ----------
    binary_name: str
        Executable to run

    Returns
    -------
    str
        Full path to the executable
    """
    bin_path = shutil.which(binary_name)
    if bin_path is None:
        if binary_name in ["fstcompile", "fstarcsort", "fstconvert"]:
            raise ThirdpartyError(binary_name, open_fst=True)
        else:
            raise ThirdpartyError(binary_name)
    if " " in bin_path:
        return f'"{bin_path}"'
    return bin_path


def log_kaldi_errors(error_logs: List[str]) -> None:
    """
    Save details of Kaldi processing errors to a logger

    Parameters
    ----------
    error_logs: list[str]
        Kaldi log files with errors
    """
    logger.debug(f"There were {len(error_logs)} kaldi processing files that had errors:")
    for path in error_logs:
        logger.debug("")
        logger.debug(path)
        with mfa_open(path, "r") as f:
            for line in f:
                logger.debug("\t" + line.strip())


def read_feats(proc: subprocess.Popen, raw_id=False) -> Dict[str, np.array]:
    """
    Inspired by https://github.com/it-muslim/kaldi-helpers/blob/master/kaldi-helpers/kaldi_io.py#L87

    Reading from stdout, import feats (or feats-like) data as a numpy array
    As feats are generated "on-fly" in kaldi, there is no a feats file
    (except most simple cases like raw mfcc, plp or fbank).  So, that is why
    we take feats as a command rather that a file path. Can be applied to
    other commands (like gmm-compute-likes) generating an output in same
    format as feats, i.e:
    utterance_id_1  [
      70.31843 -2.872698 -0.06561285 22.71824 -15.57525 ...
      78.39457 -1.907646 -1.593253 23.57921 -14.74229 ...
      ...
      57.27236 -16.17824 -15.33368 -5.945696 0.04276848 ... -0.5812851 ]
    utterance_id_2  [
      64.00951 -8.952017 4.134113 33.16264 11.09073 ...
      ...

    Parameters
    ----------
    proc : subprocess.Popen
        A process that generates features or feature-like specifications

    Returns
    -------
    feats : numpy.array
        A dict of pairs {utterance: feats}
    """
    feats = []
    # current_row = 0
    current_id = None
    for line in proc.stdout:
        line = line.decode("ascii").strip()
        if "[" in line and "]" in line:
            line = line.replace("]", "").replace("[", "").split()
            ids = line.pop(0)
            if raw_id:
                utt_id = ids
            else:
                utt_id = int(ids.split("-")[-1])
            feats = np.array([float(x) for x in line])
            yield utt_id, feats
            feats = []
            continue
        elif "[" in line:
            ids = line.strip().split()[0]
            if raw_id:
                utt_id = ids
            else:
                utt_id = int(ids.split("-")[-1])
            if current_id is None:
                current_id = utt_id
            if current_id != utt_id:
                feats = np.array(feats)
                yield current_id, feats
                feats = []
                current_id = utt_id
            continue
        if not line:
            continue
        feats.append([float(x) for x in line.replace("]", "").split()])
    if current_id is not None:
        feats = np.array(feats)
        yield current_id, feats


def parse_logs(log_directory: str) -> None:
    """
    Parse the output of a Kaldi run for any errors and raise relevant MFA exceptions

    Parameters
    ----------
    log_directory: str
        Log directory to parse

    Raises
    ------
    KaldiProcessingError
        If any log files contained error lines

    """
    error_logs = []
    for name in os.listdir(log_directory):
        log_path = os.path.join(log_directory, name)
        if os.path.isdir(log_path):
            continue
        if not name.endswith(".log"):
            continue
        with mfa_open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if "error while loading shared libraries: libopenblas.so.0" in line:
                    raise ThirdpartyError("libopenblas.so.0", open_blas=True)
                for libc_version in ["GLIBC_2.27", "GLIBCXX_3.4.20"]:
                    if libc_version in line:
                        raise ThirdpartyError(libc_version, libc=True)
                if "sox FAIL formats" in line:
                    f = line.split(" ")[-1]
                    raise ThirdpartyError(f, sox=True)
                if line.startswith("ERROR") or line.startswith("ASSERTION_FAILED"):
                    error_logs.append(log_path)
                    break
    if error_logs:
        raise KaldiProcessingError(error_logs)


class Counter(object):
    """
    Multiprocessing counter object for keeping track of progress

    Attributes
    ----------
    val: :func:`~multiprocessing.Value`
        Integer to increment
    lock: :class:`~multiprocessing.Lock`
        Lock for process safety
    """

    def __init__(self, init_val: int = 0):
        self.val = mp.Value("i", init_val)
        self.lock = mp.Lock()

    def increment(self, value=1) -> None:
        """Increment the counter"""
        with self.lock:
            self.val.value += value

    def value(self) -> int:
        """Get the current value of the counter"""
        with self.lock:
            return self.val.value


class ProgressCallback(object):
    """
    Class for sending progress indications back to the main process
    """

    def __init__(self, callback=None, total_callback=None):
        self._total = 0
        self.callback = callback
        self.total_callback = total_callback
        self._progress = 0
        self.callback_interval = 1
        self.lock = mp.Lock()
        self.start_time = None

    @property
    def total(self) -> int:
        """Total entries to process"""
        with self.lock:
            return self._total

    @property
    def progress(self) -> int:
        """Current number of entries processed"""
        with self.lock:
            return self._progress

    @property
    def progress_percent(self) -> float:
        """Current progress as percetage"""
        with self.lock:
            if not self._total:
                return 0.0
            return self._progress / self._total

    def update_total(self, total: int) -> None:
        """
        Update the total for the callback

        Parameters
        ----------
        total: int
            New total
        """
        with self.lock:
            if self._total == 0 and total != 0:
                self.start_time = time.time()
            self._total = total
            if self.total_callback is not None:
                self.total_callback(self._total)

    def set_progress(self, progress: int) -> None:
        """
        Update the number of entries processed for the callback

        Parameters
        ----------
        progress: int
            New progress
        """
        with self.lock:
            self._progress = progress

    def increment_progress(self, increment: int) -> None:
        """
        Increment the number of entries processed for the callback

        Parameters
        ----------
        increment: int
            Update the progress by this amount
        """
        with self.lock:
            self._progress += increment
            if self.callback is not None:
                current_time = time.time()
                current_duration = current_time - self.start_time
                time_per_iteration = current_duration / self._progress
                remaining_iterations = self._total - self._progress
                remaining_time = datetime.timedelta(
                    seconds=int(time_per_iteration * remaining_iterations)
                )
                self.callback(self._progress, str(remaining_time))


class Stopped(object):
    """
    Multiprocessing class for detecting whether processes should stop processing and exit ASAP

    Attributes
    ----------
    val: :func:`~multiprocessing.Value`
        0 if not stopped, 1 if stopped
    lock: :class:`~multiprocessing.Lock`
        Lock for process safety
    _source: multiprocessing.Value
        1 if it was a Ctrl+C event that stopped it, 0 otherwise
    """

    def __init__(self, initval: Union[bool, int] = False):
        self.val = mp.Value("i", initval)
        self.lock = mp.Lock()
        self._source = mp.Value("i", 0)

    def reset(self) -> None:
        """Signal that work should stop asap"""
        with self.lock:
            self.val.value = False

    def stop(self) -> None:
        """Signal that work should stop asap"""
        with self.lock:
            self.val.value = True

    def stop_check(self) -> int:
        """Check whether a process should stop"""
        with self.lock:
            return self.val.value

    def set_sigint_source(self) -> None:
        """Set the source as a ctrl+c"""
        with self.lock:
            self._source.value = True

    def source(self) -> int:
        """Get the source value"""
        with self.lock:
            return self._source.value


class ProcessWorker(mp.Process):
    """
    Multiprocessing function work

    Parameters
    ----------
    job_name: int
        Integer number of job
    job_q: :class:`~multiprocessing.Queue`
        Job queue to pull arguments from
    function: Callable
        Multiprocessing function to call on arguments from job_q
    return_dict: dict
        Dictionary for collecting errors
    stopped: :class:`~montreal_forced_aligner.utils.Stopped`
        Stop check
    return_info: dict[int, Any], optional
        Optional dictionary to fill if the function should return information to main thread
    """

    def __init__(
        self,
        job_name: int,
        job_q: mp.Queue,
        function: Callable,
        return_q: mp.Queue,
        stopped: Stopped,
    ):
        mp.Process.__init__(self)
        self.job_name = job_name
        self.function = function
        self.job_q = job_q
        self.return_q = return_q
        self.stopped = stopped
        self.finished_processing = Stopped()

    def run(self) -> None:
        """
        Run through the arguments in the queue apply the function to them
        """
        while True:
            try:
                arguments = self.job_q.get(timeout=1)
            except Empty:
                self.finished_processing.stop()
                break
            try:
                if isinstance(arguments, MfaArguments):
                    result = self.function(arguments)
                else:
                    result = self.function(*arguments)
                self.return_q.put((self.job_name, result))
            except Exception as e:
                self.stopped.stop()
                if isinstance(e, (KaldiProcessingError, MultiprocessingError)):
                    e.job_name = self.job_name
                self.return_q.put((self.job_name, e))


class KaldiProcessWorker(mp.Process):
    """
    Multiprocessing function work

    Parameters
    ----------
    job_name: int
        Integer number of job
    return_q: :class:`~multiprocessing.Queue`
        Queue for returning results
    function: KaldiFunction
        Multiprocessing function to call on arguments from job_q
    error_dict: dict
        Dictionary for collecting errors
    stopped: :class:`~montreal_forced_aligner.utils.Stopped`
        Stop check
    """

    def __init__(
        self,
        job_name: int,
        return_q: mp.Queue,
        function: KaldiFunction,
        stopped: Stopped,
    ):
        mp.Process.__init__(self)
        self.job_name = job_name
        self.function = function
        self.return_q = return_q
        self.stopped = stopped
        self.finished = Stopped()

    def run(self) -> None:
        """
        Run through the arguments in the queue apply the function to them
        """

        os.environ["OMP_NUM_THREADS"] = f"{GLOBAL_CONFIG.current_profile.blas_num_threads}"
        os.environ["OPENBLAS_NUM_THREADS"] = f"{GLOBAL_CONFIG.current_profile.blas_num_threads}"
        os.environ["MKL_NUM_THREADS"] = f"{GLOBAL_CONFIG.current_profile.blas_num_threads}"
        try:
            for result in self.function.run():
                self.return_q.put(result)
        except Exception as e:
            self.stopped.stop()
            if isinstance(e, KaldiProcessingError):
                e.job_name = self.job_name
            self.return_q.put(e)
        finally:
            self.finished.stop()


def run_kaldi_function(function, arguments, progress_callback, stopped: Stopped = None):
    if stopped is None:
        stopped = Stopped()
    if GLOBAL_CONFIG.use_mp:
        error_dict = {}
        return_queue = mp.Queue(10000)
        procs = []
        for i, args in enumerate(arguments):
            f = function(args)
            p = KaldiProcessWorker(i, return_queue, f, stopped)
            procs.append(p)
            p.start()
        while True:
            try:
                result = return_queue.get(timeout=1)
                if isinstance(result, Exception):
                    error_dict[getattr(result, "job_name", 0)] = result
                    continue
                if stopped.stop_check():
                    continue
            except Empty:
                for proc in procs:
                    if not proc.finished.stop_check():
                        break
                else:
                    break
                continue
            yield result
            progress_callback(1)
        for p in procs:
            p.join()

        if error_dict:
            for v in error_dict.values():
                raise v

    else:
        for args in arguments:
            f = function(args)
            for result in f.run():
                if stopped.stop_check():
                    break
                yield result
                progress_callback(1)


def run_non_mp(
    function: Callable,
    argument_list: List[Union[Tuple[Any, ...], MfaArguments]],
    log_directory: str,
    return_info: bool = False,
) -> Optional[Dict[Any, Any]]:
    """
    Similar to :func:`run_mp`, but no additional processes are used and the jobs are evaluated in sequential order

    Parameters
    ----------
    function: Callable
        Multiprocessing function to evaluate
    argument_list: list
        List of arguments to process
    log_directory: str
        Directory that all log information from the processes goes to
    return_info: dict, optional
        If the function returns information, supply the return dict to populate

    Returns
    -------
    dict, optional
        If the function returns information, returns the dictionary it was supplied with
    """
    if return_info:
        info = {}
        for i, args in enumerate(argument_list):
            if isinstance(args, MfaArguments):
                info[i] = function(args)
            else:
                info[i] = function(*args)
        parse_logs(log_directory)
        return info

    for args in argument_list:
        if isinstance(args, MfaArguments):
            function(args)
        else:
            function(*args)
    parse_logs(log_directory)


def run_mp(
    function: Callable,
    argument_list: List[Union[Tuple[Any, ...], MfaArguments]],
    log_directory: str,
    return_info: bool = False,
) -> Optional[Dict[int, Any]]:
    """
    Apply a function for each job in parallel

    Parameters
    ----------
    function: Callable
        Multiprocessing function to apply
    argument_list: list
        Arguments for each job
    log_directory: str
        Directory that all log information from the processes goes to
    return_info: dict, optional
        If the function returns information, supply the return dict to populate
    """

    os.environ["OMP_NUM_THREADS"] = f"{GLOBAL_CONFIG.current_profile.blas_num_threads}"
    os.environ["OPENBLAS_NUM_THREADS"] = f"{GLOBAL_CONFIG.current_profile.blas_num_threads}"
    os.environ["MKL_NUM_THREADS"] = f"{GLOBAL_CONFIG.current_profile.blas_num_threads}"
    stopped = Stopped()
    job_queue = mp.Queue()
    return_queue = mp.Queue()
    error_dict = {}
    info = {}
    for a in argument_list:
        job_queue.put(a)
    procs = []
    for i in range(len(argument_list)):
        p = ProcessWorker(i, job_queue, function, return_queue, stopped)
        procs.append(p)
        p.start()

    while True:
        try:
            job_name, result = return_queue.get(timeout=1)
            if stopped.stop_check():
                continue
        except Empty:
            for proc in procs:
                if not proc.finished_processing.stop_check():
                    break
            else:
                break
            continue
        if isinstance(result, (KaldiProcessingError, MultiprocessingError)):
            error_dict[job_name] = result
            continue
        info[job_name] = result
    for p in procs:
        p.join()
    if error_dict:
        for v in error_dict.values():
            raise v

    parse_logs(log_directory)
    if return_info:
        return info
