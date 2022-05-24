"""
Utility functions
=================

"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
from queue import Empty
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ansiwrap
import sqlalchemy
from colorama import Fore, Style
from sqlalchemy.orm import Session

from montreal_forced_aligner.abc import KaldiFunction
from montreal_forced_aligner.data import DatasetType, MfaArguments
from montreal_forced_aligner.db import Corpus, Dictionary
from montreal_forced_aligner.exceptions import (
    KaldiProcessingError,
    MultiprocessingError,
    ThirdpartyError,
)
from montreal_forced_aligner.models import MODEL_TYPES

__all__ = [
    "thirdparty_binary",
    "log_kaldi_errors",
    "guess_model_type",
    "get_mfa_version",
    "parse_logs",
    "CustomFormatter",
    "Counter",
    "Stopped",
    "ProcessWorker",
    "run_mp",
    "run_non_mp",
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


def inspect_database(path: str) -> DatasetType:
    if not os.path.exists(path):
        return DatasetType.NONE
    engine = sqlalchemy.create_engine(f"sqlite:///file:{path}?mode=ro&nolock=1&uri=true")
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


def get_class_for_dataset_type(dataset_type: DatasetType):
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
        p = subprocess.run([thirdparty_binary(fn), "--help"], capture_output=True, text=True)
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
        if binary_name in ["fstcompile", "fstarcsort", "fstconvert"] and sys.platform != "win32":
            raise ThirdpartyError(binary_name, open_fst=True)
        else:
            raise ThirdpartyError(binary_name)
    if " " in bin_path:
        return f'"{bin_path}"'
    return bin_path


def log_kaldi_errors(error_logs: List[str], logger: logging.Logger) -> None:
    """
    Save details of Kaldi processing errors to a logger

    Parameters
    ----------
    error_logs: list[str]
        Kaldi log files with errors
    logger: :class:`~logging.Logger`
        Logger to output to
    """
    logger.debug(f"There were {len(error_logs)} kaldi processing files that had errors:")
    for path in error_logs:
        logger.debug("")
        logger.debug(path)
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                logger.debug("\t" + line.strip())


def guess_model_type(path: str) -> List[str]:
    """
    Guess a model type given a path

    Parameters
    ----------
    path: str
        Model archive to guess

    Returns
    -------
    list[str]
        Possible model types that use that extension
    """
    ext = os.path.splitext(path)[1]
    if not ext:
        return []
    possible = []
    for m, mc in MODEL_TYPES.items():
        if ext in mc.extensions:
            possible.append(m)
    return possible


class CustomFormatter(logging.Formatter):
    """
    Custom log formatter class for MFA to highlight messages and incorporate terminal options from
    the global configuration
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .config import load_global_config

        config = load_global_config()
        self.width = config["terminal_width"]
        use_colors = config.get("terminal_colors", True)
        red = ""
        green = ""
        yellow = ""
        blue = ""
        reset = ""
        if use_colors:
            red = Fore.RED
            green = Fore.GREEN
            yellow = Fore.YELLOW
            blue = Fore.CYAN
            reset = Style.RESET_ALL

        self.FORMATS = {
            logging.DEBUG: (f"{blue}DEBUG{reset} - ", "%(message)s"),
            logging.INFO: (f"{green}INFO{reset} - ", "%(message)s"),
            logging.WARNING: (f"{yellow}WARNING{reset} - ", "%(message)s"),
            logging.ERROR: (f"{red}ERROR{reset} - ", "%(message)s"),
            logging.CRITICAL: (f"{red}CRITICAL{reset} - ", "%(message)s"),
        }

    def format(self, record: logging.LogRecord):
        """
        Format a given log message

        Parameters
        ----------
        record: logging.LogRecord
            Log record to format

        Returns
        -------
        str
            Formatted log message
        """
        log_fmt = self.FORMATS.get(record.levelno)
        return ansiwrap.fill(
            record.getMessage(),
            initial_indent=log_fmt[0],
            subsequent_indent=" " * len(log_fmt[0]),
            width=self.width,
        )


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
        with open(log_path, "r", encoding="utf8") as f:
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
        from .config import BLAS_THREADS

        os.environ["OPENBLAS_NUM_THREADS"] = f"{BLAS_THREADS}"
        os.environ["MKL_NUM_THREADS"] = f"{BLAS_THREADS}"
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
    from .config import BLAS_THREADS

    os.environ["OPENBLAS_NUM_THREADS"] = f"{BLAS_THREADS}"
    os.environ["MKL_NUM_THREADS"] = f"{BLAS_THREADS}"
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
