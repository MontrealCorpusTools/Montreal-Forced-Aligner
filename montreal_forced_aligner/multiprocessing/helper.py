from __future__ import annotations
from typing import TYPE_CHECKING, Union, Callable, Dict, Optional, List, Any, Tuple
import multiprocessing as mp
from queue import Empty
import traceback
import os
import sys

from ..utils import parse_logs


class Counter(object):
    def __init__(self, init_val: int=0):
        self.val = mp.Value('i', init_val)
        self.lock = mp.Lock()

    def increment(self) -> None:
        with self.lock:
            self.val.value += 1

    def value(self) -> int:
        with self.lock:
            return self.val.value


class Stopped(object):
    def __init__(self, initval: Union[bool, int]=False):
        self.val = mp.Value('i', initval)
        self.lock = mp.Lock()
        self._source = mp.Value('i', 0)

    def stop(self) -> None:
        with self.lock:
            self.val.value = True

    def stop_check(self) -> int:
        with self.lock:
            return self.val.value

    def set_sigint_source(self) -> None:
        with self.lock:
            self._source.value = True

    def source(self) -> int:
        with self.lock:
            return self._source.value


class ProcessWorker(mp.Process):
    def __init__(self, job_name: int, job_q: mp.Queue, function: Callable,
                 return_dict: Dict, stopped: Stopped, return_info: Optional[Dict[str, Any]] = None):
        mp.Process.__init__(self)
        self.job_name = job_name
        self.function = function
        self.job_q = job_q
        self.return_dict = return_dict
        self.return_info = return_info
        self.stopped = stopped

    def run(self) -> None:
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        while True:
            try:
                arguments = self.job_q.get(timeout=1)
            except Empty as error:
                break
            self.job_q.task_done()
            if self.stopped.stop_check():
                continue
            try:
                result = self.function(*arguments)
                if self.return_info is not None:
                    self.return_info[self.job_name] = result
            except Exception as e:
                self.stopped.stop()
                self.return_dict['error'] = arguments, Exception(traceback.format_exception(*sys.exc_info()))
        return


def run_non_mp(function: Callable,
               argument_list: List[Tuple[Any, ...]],
               log_directory: str,
               return_info: Optional[Dict[str, Any]]=None) -> Optional[Dict[Any, Any]]:
    if return_info is not None:
        for i, args in enumerate(argument_list):
            return_info[i] = function(*args)
        parse_logs(log_directory)
        return return_info

    for args in argument_list:
        function(*args)
    parse_logs(log_directory)


def run_mp(function: Callable,
           argument_list: List[Tuple[Any, ...]],
           log_directory: str,
           return_info: Optional[Dict[str, Any]]=None) -> None:  # pragma: no cover
    stopped = Stopped()
    manager = mp.Manager()
    job_queue = manager.Queue()
    return_dict = manager.dict()
    for a in argument_list:
        job_queue.put(a, False)
    procs = []
    for i in range(len(argument_list)):
        p = ProcessWorker(i, job_queue, function, return_dict, stopped, return_info)
        procs.append(p)
        p.start()

    for p in procs:
        p.join()
    if 'error' in return_dict:
        element, exc = return_dict['error']
        raise exc

    parse_logs(log_directory)
