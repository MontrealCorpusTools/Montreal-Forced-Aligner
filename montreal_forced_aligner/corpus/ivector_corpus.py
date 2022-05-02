"""Classes for corpora that use ivectors as features"""
import multiprocessing as mp
import os
import time
from queue import Empty
from typing import List

import tqdm

from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpusMixin
from montreal_forced_aligner.corpus.features import (
    ExtractIvectorsArguments,
    ExtractIvectorsFunction,
    IvectorConfigMixin,
)
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.utils import KaldiProcessWorker, Stopped

__all__ = ["IvectorCorpusMixin"]


class IvectorCorpusMixin(AcousticCorpusMixin, IvectorConfigMixin):
    """
    Abstract corpus mixin for corpora that extract ivectors

    See Also
    --------
    :class:`~montreal_forced_aligner.corpus.acoustic_corpus.AcousticCorpusMixin`
        For dictionary and corpus parsing parameters
    :class:`~montreal_forced_aligner.corpus.features.IvectorConfigMixin`
        For ivector extraction parameters

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def ie_path(self):
        """Ivector extractor ie path"""
        raise NotImplementedError

    @property
    def dubm_path(self):
        """DUBM model path"""
        raise

    def extract_ivectors_arguments(self) -> List[ExtractIvectorsArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.corpus.features.ExtractIvectorsFunction`

        Returns
        -------
        list[ExtractIvectorsArguments]
            Arguments for processing
        """
        return [
            ExtractIvectorsArguments(
                j.name,
                getattr(self, "db_path", ""),
                os.path.join(self.working_log_directory, f"extract_ivectors.{j.name}.log"),
                j.construct_path(self.split_directory, "feats", "scp"),
                self.ivector_options,
                self.ie_path,
                j.construct_path(self.split_directory, "ivectors", "scp"),
                self.model_path,
                self.dubm_path,
            )
            for j in self.jobs
        ]

    def extract_ivectors(self) -> None:
        """
        Multiprocessing function that extracts job_name-vectors.

        See Also
        --------
        :class:`~montreal_forced_aligner.corpus.features.ExtractIvectorsFunction`
            Multiprocessing helper function for each job
        :meth:`.IvectorCorpusMixin.extract_ivectors_arguments`
            Job method for generating arguments for helper function
        :kaldi_steps_sid:`extract_ivectors`
            Reference Kaldi script
        """
        begin = time.time()

        log_dir = self.working_log_directory
        os.makedirs(log_dir, exist_ok=True)

        arguments = self.extract_ivectors_arguments()
        with tqdm.tqdm(total=self.num_speakers, disable=getattr(self, "quiet", False)) as pbar:
            if self.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(arguments):
                    function = ExtractIvectorsFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    pbar.update(1)
                    if isinstance(result, KaldiProcessingError):
                        error_dict[result.job_name] = result
                        continue
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in arguments:
                    function = ExtractIvectorsFunction(args)
                    for _ in function.run():
                        pbar.update(1)
        self.log_debug(f"Ivector extraction took {time.time() - begin}")
