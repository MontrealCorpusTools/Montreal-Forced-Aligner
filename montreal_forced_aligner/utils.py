"""
Utility functions
=================

"""
from __future__ import annotations

import datetime
import logging
import multiprocessing as mp
import os
import pathlib
import queue
import re
import shutil
import subprocess
import threading
import time
import typing
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

import sqlalchemy
from sqlalchemy.orm import Session
from tqdm.rich import tqdm

from montreal_forced_aligner import config
from montreal_forced_aligner.abc import KaldiFunction
from montreal_forced_aligner.data import CtmInterval, DatasetType
from montreal_forced_aligner.db import Corpus, Dictionary
from montreal_forced_aligner.exceptions import (
    DictionaryError,
    KaldiProcessingError,
    ThirdpartyError,
)
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.textgrid import process_ctm_line

__all__ = [
    "check_third_party",
    "thirdparty_binary",
    "log_kaldi_errors",
    "get_mfa_version",
    "parse_logs",
    "inspect_database",
    "Counter",
    "ProgressCallback",
    "KaldiProcessWorker",
    "parse_ctm_output",
    "run_kaldi_function",
    "thread_logger",
    "parse_dictionary_file",
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

    string = f"postgresql+psycopg2://@/{name}?host={config.database_socket()}"
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
    typing.Union[None, AcousticCorpus, TextCorpus, AcousticCorpusWithPronunciations, DictionaryTextCorpus,MultispeakerDictionary]
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
    path: Path,
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
    path: :class:`~pathlib.Path`
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
    prob_pattern = re.compile(r"\b(\d+\.\d+|1)\b")
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
            word = unicodedata.normalize("NFKC", word)
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
        __version__ = "3.0.0"
    return __version__


def check_third_party():
    """
    Checks whether third party software is available on the path

    Raises
    -------
    :class:`~montreal_forced_aligner.exceptions.ThirdpartyError`
    """
    bin_path = shutil.which("initdb")
    if bin_path is None and config.USE_POSTGRES:
        raise ThirdpartyError("initdb (for postgresql)")
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


def parse_logs(log_directory: Path) -> None:
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
    for log_path in log_directory.iterdir():
        if log_path.is_dir():
            continue
        if log_path.suffix != ".log":
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
    lock: :class:`~threading.Lock`
        Lock for threading safety
    """

    def __init__(
        self,
    ):
        self._value = 0
        self.lock = threading.Lock()

    def increment(self, value=1) -> None:
        """Increment the counter"""
        with self.lock:
            self._value += value

    def value(self) -> int:
        """Get the current value of the counter"""
        with self.lock:
            return self._value


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
        self.lock = threading.Lock()
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


class KaldiProcessWorker(threading.Thread):
    """
    Multiprocessing function work

    Parameters
    ----------
    job_name: int
        Integer number of job
    return_q: :class:`~queue.Queue`
        Queue for returning results
    function: KaldiFunction
        Multiprocessing function to call on arguments from job_q
    stopped: :class:`~threading.Event`
        Stop check
    """

    def __init__(
        self,
        job_name: int,
        return_q: typing.Union[mp.Queue, queue.Queue],
        function: KaldiFunction,
        stopped: threading.Event,
    ):
        super().__init__(name=str(job_name))
        self.job_name = job_name
        self.function = function
        self.function.callback = self.add_to_return_queue
        self.return_q = return_q
        self.stopped = stopped
        self.finished = threading.Event()

    def add_to_return_queue(self, result):
        if self.stopped.is_set():
            return
        self.return_q.put(result)

    def run(self) -> None:
        """
        Run through the arguments in the queue apply the function to them
        """

        os.environ["OMP_NUM_THREADS"] = f"{config.BLAS_NUM_THREADS}"
        os.environ["OPENBLAS_NUM_THREADS"] = f"{config.BLAS_NUM_THREADS}"
        os.environ["MKL_NUM_THREADS"] = f"{config.BLAS_NUM_THREADS}"
        try:
            self.function.run()
        except Exception as e:
            self.stopped.set()
            if isinstance(e, KaldiProcessingError):
                e.job_name = self.job_name
            self.return_q.put(e)
        finally:
            self.finished.set()


class KaldiProcessWorkerMp(mp.Process):
    """
    Multiprocessing function work

    Parameters
    ----------
    job_name: int
        Integer number of job
    return_q: :class:`~queue.Queue`
        Queue for returning results
    function: KaldiFunction
        Multiprocessing function to call on arguments from job_q
    stopped: :class:`~threading.Event`
        Stop check
    """

    def __init__(
        self,
        job_name: int,
        return_q: mp.Queue,
        function: KaldiFunction,
        stopped: mp.Event,
    ):
        super().__init__(name=str(job_name))
        self.job_name = job_name
        self.function = function
        self.function.callback = self.add_to_return_queue
        self.return_q = return_q
        self.stopped = stopped
        self.finished = mp.Event()

    def add_to_return_queue(self, result):
        if self.stopped.is_set():
            return
        self.return_q.put(result)

    def run(self) -> None:
        """
        Run through the arguments in the queue apply the function to them
        """

        os.environ["OMP_NUM_THREADS"] = f"{config.BLAS_NUM_THREADS}"
        os.environ["OPENBLAS_NUM_THREADS"] = f"{config.BLAS_NUM_THREADS}"
        os.environ["MKL_NUM_THREADS"] = f"{config.BLAS_NUM_THREADS}"
        try:
            self.function.run()
        except Exception as e:
            self.stopped.set()
            if isinstance(e, KaldiProcessingError):
                e.job_name = self.job_name
            self.return_q.put(e)
        finally:
            self.finished.set()


@contextmanager
def thread_logger(
    log_name: str, log_path: typing.Union[pathlib.Path, str], job_name: int = None
) -> logging.Logger:
    kalpy_logging = logging.getLogger(log_name)
    file_handler = logging.FileHandler(log_path, encoding="utf8")
    file_handler.setLevel(logging.DEBUG)
    if config.USE_THREADING:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(thread)d - %(threadName)s - %(levelname)s - %(message)s"
        )
        if job_name is None:
            log_filter = IgnoreThreadsFilter()
        else:
            log_filter = ThreadFilter(job_name)
        file_handler.addFilter(log_filter)
    else:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    try:
        kalpy_logging.addHandler(file_handler)
        yield kalpy_logging
    finally:
        file_handler.close()
        kalpy_logging.removeHandler(file_handler)


class ThreadFilter(logging.Filter):
    """Only accept log records from a specific thread or thread name"""

    def __init__(self, thread_name):
        self._thread_name = str(thread_name)

    def filter(self, record):
        if self._thread_name is not None and record.threadName != self._thread_name:
            return False
        return True


class IgnoreThreadsFilter(logging.Filter):
    """Only accepts log records that originated from the main thread"""

    def __init__(self):
        self._main_thread_id = threading.main_thread().ident

    def filter(self, record):
        return record.thread == self._main_thread_id


def run_kaldi_function(
    function, arguments, stopped: threading.Event = None, total_count: int = None
):
    if config.USE_THREADING:
        Event = threading.Event
        Queue = queue.Queue
        Worker = KaldiProcessWorker
    else:
        Event = mp.Event
        Queue = mp.Queue
        Worker = KaldiProcessWorkerMp
    if stopped is None:
        stopped = Event()
    error_dict = {}
    return_queue = Queue(10000)
    callback_interval = 10
    num_done = 0
    last_update = 0
    pbar = None
    progress_callback = None
    if not config.QUIET and total_count:
        pbar = tqdm(total=total_count, maxinterval=0)
        progress_callback = pbar.update
    update_time = time.time()
    if config.USE_MP:
        procs = []
        for args in arguments:
            f = function(args)
            p = Worker(args.job_name, return_queue, f, stopped)
            procs.append(p)
            p.start()
        try:
            while True:
                try:
                    result = return_queue.get(timeout=1)
                    if isinstance(result, Exception):
                        error_dict[getattr(result, "job_name", 0)] = result
                        stopped.set()
                        continue
                    if stopped.is_set():
                        continue
                    yield result
                    if progress_callback is not None:
                        if isinstance(result, int):
                            num_done += result
                        else:
                            num_done += 1
                        if time.time() - update_time >= callback_interval:
                            if num_done - last_update > 0:
                                progress_callback(num_done - last_update)
                                last_update = num_done
                            update_time = time.time()
                    if isinstance(return_queue, queue.Queue):
                        return_queue.task_done()
                except queue.Empty:
                    for proc in procs:
                        if not proc.finished.is_set():
                            break
                    else:
                        break
                    continue
                except Exception as e:
                    if isinstance(e, KeyboardInterrupt):
                        logger.debug("Received ctrl+c event")
                    stopped.set()
                    error_dict["main_thread"] = e
                    import sys
                    import traceback

                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    logger.debug(
                        "\n".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                    )
                    continue

        finally:
            for p in procs:
                p.join()
                del p.function
            del procs
            del return_queue
            del stopped
            del arguments

        if error_dict:
            for v in error_dict.values():
                raise v
    else:
        for args in arguments:
            f = function(args)
            p = Worker(args.job_name, return_queue, f, stopped)
            p.start()
            try:
                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if isinstance(result, Exception):
                            error_dict[getattr(result, "job_name", 0)] = result
                            stopped.set()
                            continue
                        if stopped.is_set():
                            continue
                        yield result
                        if progress_callback is not None:
                            if isinstance(result, int):
                                num_done += result
                            else:
                                num_done += 1
                            if num_done - last_update >= callback_interval:
                                progress_callback(num_done - last_update)
                                last_update = num_done
                        if isinstance(return_queue, queue.Queue):
                            return_queue.task_done()
                    except queue.Empty:
                        if not p.finished.is_set():
                            continue
                        else:
                            break
                    except (KeyboardInterrupt, SystemExit):
                        logger.debug("Received ctrl+c event")
                        stopped.set()
                        continue
                    except Exception as e:
                        if isinstance(e, KeyboardInterrupt):
                            logger.debug("Received ctrl+c event")
                        stopped.set()
                        error_dict["main_thread"] = e
                        import sys
                        import traceback

                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        logger.debug(
                            "\n".join(
                                traceback.format_exception(exc_type, exc_value, exc_traceback)
                            )
                        )
                        continue

            finally:
                p.join()

        if error_dict:
            for v in error_dict.values():
                raise v
    if pbar is not None and num_done > last_update:
        progress_callback(num_done - last_update)
        pbar.refresh()
        pbar.close()
