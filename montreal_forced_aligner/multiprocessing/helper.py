"""
Multiprocessing helpers
-----------------------

"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import traceback
from queue import Empty
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..utils import parse_logs

__all__ = ["Counter", "Stopped", "ProcessWorker", "run_mp", "run_non_mp"]


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

    def increment(self) -> None:
        """Increment the counter"""
        with self.lock:
            self.val.value += 1

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
    return_dict: Dict
        Dictionary for collecting errors
    stopped: :class:`~montreal_forced_aligner.multiprocessing.helper.Stopped`
        Stop check
    return_info: Dict[int, Any], optional
        Optional dictionary to fill if the function should return information to main thread
    """

    def __init__(
        self,
        job_name: int,
        job_q: mp.Queue,
        function: Callable,
        return_dict: Dict,
        stopped: Stopped,
        return_info: Optional[Dict[int, Any]] = None,
    ):
        mp.Process.__init__(self)
        self.job_name = job_name
        self.function = function
        self.job_q = job_q
        self.return_dict = return_dict
        self.return_info = return_info
        self.stopped = stopped

    def run(self) -> None:
        """
        Run through the arguments in the queue apply the function to them
        """
        try:
            arguments = self.job_q.get(timeout=1)
        except Empty:
            return
        self.job_q.task_done()
        try:
            result = self.function(*arguments)
            if self.return_info is not None:
                self.return_info[self.job_name] = result
        except Exception:
            self.stopped.stop()
            self.return_dict["error"] = arguments, Exception(
                traceback.format_exception(*sys.exc_info())
            )


def run_non_mp(
    function: Callable,
    argument_list: List[Tuple[Any, ...]],
    log_directory: str,
    return_info: bool = False,
) -> Optional[Dict[Any, Any]]:
    """
    Similar to run_mp, but no additional processes are used and the jobs are evaluated in sequential order

    Parameters
    ----------
    function: Callable
        Multiprocessing function to evaluate
    argument_list: List
        List of arguments to process
    log_directory: str
        Directory that all log information from the processes goes to
    return_info: Dict, optional
        If the function returns information, supply the return dict to populate

    Returns
    -------
    Dict, optional
        If the function returns information, returns the dictionary it was supplied with
    """
    if return_info:
        info = {}
        for i, args in enumerate(argument_list):
            info[i] = function(*args)
        parse_logs(log_directory)
        return info

    for args in argument_list:
        function(*args)
    parse_logs(log_directory)


def run_mp(
    function: Callable,
    argument_list: List[Tuple[Any, ...]],
    log_directory: str,
    return_info: bool = False,
) -> Optional[Dict[int, Any]]:
    """
    Apply a function for each job in parallel

    Parameters
    ----------
    function: Callable
        Multiprocessing function to apply
    argument_list: List
        List of arguments for each job
    log_directory: str
        Directory that all log information from the processes goes to
    return_info: Dict, optional
        If the function returns information, supply the return dict to populate
    """
    from ..config import BLAS_THREADS

    os.environ["OPENBLAS_NUM_THREADS"] = f"{BLAS_THREADS}"
    os.environ["MKL_NUM_THREADS"] = f"{BLAS_THREADS}"
    stopped = Stopped()
    manager = mp.Manager()
    job_queue = manager.Queue()
    return_dict = manager.dict()
    info = None
    if return_info:
        info = manager.dict()
    for a in argument_list:
        job_queue.put(a)
    procs = []
    for i in range(len(argument_list)):
        p = ProcessWorker(i, job_queue, function, return_dict, stopped, info)
        procs.append(p)
        p.start()

    for p in procs:
        p.join()
    if "error" in return_dict:
        _, exc = return_dict["error"]
        raise exc

    parse_logs(log_directory)
    if return_info:
        return info
