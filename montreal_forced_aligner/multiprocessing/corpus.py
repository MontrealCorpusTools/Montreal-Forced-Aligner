"""
Corpus loading worker
---------------------


"""
from __future__ import annotations

import multiprocessing as mp
import sys
import traceback
from queue import Empty
from typing import TYPE_CHECKING, Dict, Union

from ..exceptions import TextGridParseError, TextParseError

if TYPE_CHECKING:
    from ..corpus import OneToManyMappingType, OneToOneMappingType
    from ..corpus.base import SoundFileInfoDict

    FileInfoDict = Dict[
        str, Union[str, SoundFileInfoDict, OneToOneMappingType, OneToManyMappingType]
    ]
    from .helper import Stopped


__all__ = ["CorpusProcessWorker"]


class CorpusProcessWorker(mp.Process):
    """
    Multiprocessing corpus loading worker

    Attributes
    ----------
    job_q: :class:`~multiprocessing.Queue`
        Job queue for files to process
    return_dict: Dict
        Dictionary to catch errors
    return_q: :class:`~multiprocessing.Queue`
        Return queue for processed Files
    stopped: :func:`~montreal_forced_aligner.multiprocessing.helper.Stopped`
        Stop check for whether corpus loading should exit
    finished_adding: :class:`~montreal_forced_aligner.multiprocessing.helper.Stopped`
        Signal that the main thread has stopped adding new files to be processed
    """

    def __init__(
        self,
        job_q: mp.Queue,
        return_dict: Dict,
        return_q: mp.Queue,
        stopped: Stopped,
        finished_adding: Stopped,
    ):
        mp.Process.__init__(self)
        self.job_q = job_q
        self.return_dict = return_dict
        self.return_q = return_q
        self.stopped = stopped
        self.finished_adding = finished_adding

    def run(self) -> None:
        """
        Run the corpus loading job
        """
        from ..corpus.classes import parse_file

        while True:
            try:
                arguments = self.job_q.get(timeout=1)
            except Empty:
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
                self.return_dict["decode_error_files"].append(e)
            except TextGridParseError as e:
                self.return_dict["textgrid_read_errors"][e.file_name] = e
            except Exception:
                self.stopped.stop()
                self.return_dict["error"] = arguments, Exception(
                    traceback.format_exception(*sys.exc_info())
                )
        return
