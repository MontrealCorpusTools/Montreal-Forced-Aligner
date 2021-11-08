"""Class definitions for Segmentation based on voice activity in MFA"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .corpus import TranscribeCorpus
    from .config import SegmentationConfig, ConfigDict
    from logging import Logger

import os
import shutil
from decimal import Decimal

from praatio import textgrid

from .config import TEMP_DIR
from .exceptions import KaldiProcessingError
from .multiprocessing.ivector import segment_vad
from .utils import log_kaldi_errors, parse_logs

SegmentationType = List[Dict[str, float]]

__all__ = ["Segmenter"]


class Segmenter(object):
    """
    Class for performing speaker classification

    Parameters
    ----------
    corpus : :class:`~montreal_forced_aligner.corpus.TranscribeCorpus`
        Corpus object for the dataset
    segmentation_config : :class:`~montreal_forced_aligner.config.SegmentationConfig`
        Configuration for alignment
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``
    debug : bool
        Flag for running in debug mode, defaults to false
    verbose : bool
        Flag for running in verbose mode, defaults to false
    logger : logging.Logger, optional
        Logger to use
    """

    def __init__(
        self,
        corpus: TranscribeCorpus,
        segmentation_config: SegmentationConfig,
        temp_directory: Optional[str] = None,
        debug: Optional[bool] = False,
        verbose: Optional[bool] = False,
        logger: Optional[Logger] = None,
    ):
        self.corpus = corpus
        self.segmentation_config = segmentation_config

        if not temp_directory:
            temp_directory = TEMP_DIR
        self.temp_directory = temp_directory
        self.debug = debug
        self.verbose = verbose
        self.logger = logger
        self.uses_cmvn = False
        self.uses_slices = False
        self.uses_vad = False
        self.speaker_independent = True
        self.setup()

    @property
    def segmenter_directory(self) -> str:
        """Temporary directory for segmentation"""
        return os.path.join(self.temp_directory, "segmentation")

    @property
    def vad_options(self) -> ConfigDict:
        """Options for performing VAD"""
        return {
            "energy_threshold": self.segmentation_config.energy_threshold,
            "energy_mean_scale": self.segmentation_config.energy_mean_scale,
        }

    @property
    def use_mp(self) -> bool:
        """Flag for whether to use multiprocessing"""
        return self.segmentation_config.use_mp

    def setup(self) -> None:
        """
        Sets up the corpus and segmenter for performing VAD

        Raises
        ------
        KaldiProcessingError
            If there were any errors in running Kaldi binaries
        """
        done_path = os.path.join(self.segmenter_directory, "done")
        if os.path.exists(done_path):
            self.logger.info("Classification already done, skipping initialization.")
            return
        dirty_path = os.path.join(self.segmenter_directory, "dirty")
        if os.path.exists(dirty_path):
            shutil.rmtree(self.segmenter_directory)
        log_dir = os.path.join(self.segmenter_directory, "log")
        os.makedirs(log_dir, exist_ok=True)
        try:
            self.corpus.initialize_corpus(None, self.segmentation_config.feature_config)
        except Exception as e:
            with open(dirty_path, "w"):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise

    def segment(self) -> None:
        """
        Performs VAD and segmentation into utterances

        Raises
        ------
        KaldiProcessingError
            If there were any errors in running Kaldi binaries
        """
        log_directory = os.path.join(self.segmenter_directory, "log")
        dirty_path = os.path.join(self.segmenter_directory, "dirty")
        done_path = os.path.join(self.segmenter_directory, "done")
        if os.path.exists(done_path):
            self.logger.info("Classification already done, skipping.")
            return
        try:
            self.corpus.compute_vad()
            self.uses_vad = True
            segment_vad(self)
            parse_logs(log_directory)
        except Exception as e:
            with open(dirty_path, "w"):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise
        with open(done_path, "w"):
            pass

    def export_segments(self, output_directory: str) -> None:
        """
        Export the results of segmentation as TextGrids

        Parameters
        ----------
        output_directory: str
            Directory to save segmentation TextGrids
        """
        file_dict = {}
        for segment in self.corpus.vad_segments.values():
            filename, utt_begin, utt_end = segment
            utt_begin = Decimal(utt_begin)
            utt_end = Decimal(utt_end)
            if filename not in file_dict:
                file_dict[filename] = {}
            speaker = "segments"
            text = "speech"
            if speaker not in file_dict[filename]:
                file_dict[filename][speaker] = []
            file_dict[filename][speaker].append([utt_begin, utt_end, text])
        for filename, speaker_dict in file_dict.items():
            file = self.corpus.files[filename]
            output_path = file.construct_output_path(output_directory)
            max_time = file.duration
            tg = textgrid.Textgrid()
            tg.minTimestamp = 0
            tg.maxTimestamp = max_time
            for speaker in sorted(speaker_dict.keys()):
                words = speaker_dict[speaker]
                entry_list = []
                for w in words:
                    if w[1] > max_time:
                        w[1] = max_time
                    entry_list.append(w)
                tier = textgrid.IntervalTier(speaker, entry_list, minT=0, maxT=max_time)
                tg.addTier(tier)
            tg.save(
                output_path,
                includeBlankSpaces=True,
                format="long_textgrid",
            )
