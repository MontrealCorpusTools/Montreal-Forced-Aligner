import multiprocessing as mp
from queue import Empty
import traceback
import sys

from ..helper import parse_logs, thirdparty_binary, make_path_safe


class Counter(object):
    def __init__(self, initval=0):
        self.val = mp.Value('i', initval)
        self.lock = mp.Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value


class Stopped(object):
    def __init__(self, initval=False):
        self.val = mp.Value('i', initval)
        self.lock = mp.Lock()

    def stop(self):
        with self.lock:
            self.val.value = True

    def stop_check(self):
        with self.lock:
            return self.val.value


class ProcessWorker(mp.Process):
    def __init__(self, job_q, function, return_dict, stopped):
        mp.Process.__init__(self)
        self.function = function
        self.job_q = job_q
        self.return_dict = return_dict
        self.stopped = stopped

    def run(self):
        while True:
            try:
                arguments = self.job_q.get(timeout=1)
            except Empty as error:
                break
            self.job_q.task_done()
            if self.stopped.stop_check():
                continue
            try:
                _ = self.function(*arguments)
            except Exception as e:
                self.stopped.stop()
                self.return_dict['error'] = arguments, Exception(traceback.format_exception(*sys.exc_info()))
        return


def run_non_mp(function, argument_list, log_directory):
    for args in argument_list:
        function(*args)

    parse_logs(log_directory)


def run_mp(function, argument_list, log_directory):  # pragma: no cover
    stopped = Stopped()
    manager = mp.Manager()
    job_queue = manager.Queue()
    return_dict = manager.dict()
    for a in argument_list:
        job_queue.put(a, False)
    procs = []
    for i in range(len(argument_list)):
        p = ProcessWorker(job_queue, function, return_dict, stopped)
        procs.append(p)
        p.start()

    for p in procs:
        p.join()
    if 'error' in return_dict:
        element, exc = return_dict['error']
        print(element)
        raise exc

    parse_logs(log_directory)
