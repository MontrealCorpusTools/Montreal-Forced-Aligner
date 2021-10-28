from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union
if TYPE_CHECKING:
    SpeakerCharacterType = Union[str, int]
    from logging import Logger
    from ..dictionary import DictionaryType
    from ..config import FeatureConfig
    from ..config import SegmentationConfig
import os
import time
from ..utils import log_kaldi_errors
from ..exceptions import KaldiProcessingError


from .base import BaseCorpus
from ..helper import load_scp
from ..dictionary import MultispeakerDictionary

from ..exceptions import CorpusError
from ..multiprocessing import segment_vad

from ..multiprocessing.helper import Stopped


class TranscribeCorpus(BaseCorpus):
    def __init__(self, directory: str, output_directory: str,
                 speaker_characters: SpeakerCharacterType=0,
                 num_jobs: int=3, sample_rate: int=16000, debug: bool=False, logger: Optional[Logger]=None, use_mp: bool=True,
                 no_speakers: bool=False,
                 ignore_transcriptions: bool=False, audio_directory: Optional[str]=None, skip_load: bool=False):
        super(TranscribeCorpus, self).__init__(directory, output_directory,
                                               speaker_characters,
                                               num_jobs, sample_rate, debug, logger, use_mp,
                                               audio_directory=audio_directory, skip_load=skip_load)
        self.no_speakers = no_speakers
        self.ignore_transcriptions = ignore_transcriptions
        self.vad_segments = {}
        if self.use_mp:
            self.stopped = Stopped()
        else:
            self.stopped = False
        if not self.skip_load:
            self.load()

    def load(self) -> None:
        loaded = self._load_from_temp()
        if not loaded:
            if self.use_mp:
                self.logger.debug('Loading from source with multiprocessing')
                self._load_from_source_mp()
            else:
                self.logger.debug('Loading from source without multiprocessing')
                self._load_from_source()
        else:
            self.logger.debug('Successfully loaded from temporary files')
        self.initialize_jobs()

    def initialize_corpus(self, dictionary: Optional[DictionaryType]=None, feature_config: Optional[FeatureConfig]=None) -> None:
        if not self.files:
            raise CorpusError('There were no wav files found for transcribing this corpus. Please validate the corpus.')
        self.write()
        if dictionary is not None:
            for speaker in self.speakers.values():
                speaker.set_dictionary(dictionary.get_dictionary(speaker.name))
        self.initialize_jobs()
        for j in self.jobs:
            j.set_feature_config(feature_config)
        self.feature_config = feature_config
        self.write()
        self.split()
        if self.feature_config is not None:
            try:
                self.generate_features()
            except Exception as e:
                if isinstance(e, KaldiProcessingError):
                    log_kaldi_errors(e.error_logs, self.logger)
                    e.update_log_file(self.logger.handlers[0].baseFilename)
                raise