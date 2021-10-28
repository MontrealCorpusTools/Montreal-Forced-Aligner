from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, List, Dict, Callable
if TYPE_CHECKING:
    from ..corpus import OneToOneMappingType, OneToManyMappingType
    from ..corpus.base import SoundFileInfoDict

    FileInfoDict = Dict[str, Union[str, SoundFileInfoDict, OneToOneMappingType, OneToManyMappingType]]
    from .helper import Stopped
import multiprocessing as mp
from queue import Empty
import traceback
import sys
from ..corpus.classes import parse_file
from ..exceptions import TextParseError, TextGridParseError



class CorpusProcessWorker(mp.Process):
    def __init__(self, job_q: mp.Queue, return_dict: Dict, return_q: mp.Queue, stopped: Stopped, finished_adding: Stopped):
        mp.Process.__init__(self)
        self.job_q = job_q
        self.return_dict = return_dict
        self.return_q = return_q
        self.stopped = stopped
        self.finished_adding = finished_adding

    def run(self) -> None:
        while True:
            try:
                arguments = self.job_q.get(timeout=1)
            except Empty as error:
                if self.finished_adding.stop_check():
                    break
                continue
            self.job_q.task_done()
            if self.stopped.stop_check():
                continue
            try:
                file = parse_file(*arguments, stop_check=self.stopped.stop_check)
                self.return_q.put(file)
            except TextParseError as e:
                self.return_dict['decode_error_files'].append(e)
            except TextGridParseError as e:
                self.return_dict['textgrid_read_errors'][e.file_name] = e
            except Exception as e:
                self.stopped.stop()
                self.return_dict['error'] = arguments, Exception(traceback.format_exception(*sys.exc_info()))
        return
