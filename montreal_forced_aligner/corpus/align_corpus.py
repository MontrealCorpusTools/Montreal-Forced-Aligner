from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Generator
if TYPE_CHECKING:
    from logging import Logger
    from ..dictionary import Dictionary, MultispeakerDictionary
    from ..config import FeatureConfig
    DictionaryType = Union[Dictionary, MultispeakerDictionary]
import os
import random
from ..utils import log_kaldi_errors
from ..exceptions import CorpusError, KaldiProcessingError

from .base import BaseCorpus


class AlignableCorpus(BaseCorpus):
    """
    Class that stores information about the dataset to align.

    Corpus objects have a number of mappings from either utterances or speakers
    to various properties, and mappings between utterances and speakers.

    See http://kaldi-asr.org/doc/data_prep.html for more information about
    the files that are created by this class.


    Parameters
    ----------
    directory : str
        Directory of the dataset to align
    output_directory : str
        Directory to store generated data for the Kaldi binaries
    speaker_characters : int, optional
        Number of characters in the filenames to count as the speaker ID,
        if not specified, speaker IDs are generated from directory names
    num_jobs : int, optional
        Number of processes to use, defaults to 3

    Raises
    ------
    CorpusError
        Raised if the specified corpus directory does not exist
    SampleRateError
        Raised if the wav files in the dataset do not share a consistent sample rate

    """

    def __init__(self, directory: str, output_directory: str,
                 speaker_characters: Union[int, str]=0,
                 num_jobs: int=3, sample_rate: int=16000, debug: bool=False, logger: Optional[Logger]=None, use_mp: bool=True,
                 punctuation: Optional[str]=None, clitic_markers: Optional[str]=None, parse_text_only_files: bool=False,
                 audio_directory: Optional[str]=None,
                 skip_load: bool=False):
        super(AlignableCorpus, self).__init__(directory, output_directory,
                                              speaker_characters,
                                              num_jobs, sample_rate, debug, logger, use_mp,
                                              punctuation, clitic_markers, audio_directory=audio_directory,
                                              skip_load=skip_load, parse_text_only_files=parse_text_only_files)

        if not self.skip_load:
            self.load()

    def load(self) -> None:
        self.loaded_from_temp = self._load_from_temp()
        if not self.loaded_from_temp:
            if self.use_mp:
                self.logger.debug('Loading from source with multiprocessing')
                self._load_from_source_mp()
            else:
                self.logger.debug('Loading from source without multiprocessing')
                self._load_from_source()
        else:
            self.logger.debug('Successfully loaded from temporary files')
        self.check_warnings()

    def delete_utterance(self, utterance) -> None:
        super(AlignableCorpus, self).delete_utterance(utterance)

    def check_warnings(self) -> None:
        self.issues_check = self.no_transcription_files or \
                            self.textgrid_read_errors or self.decode_error_files

    def normalized_text_iter(self, dictionary: Optional[DictionaryType]=None, min_count: int=1) -> Generator:
        unk_words = {k for k, v in self.word_counts.items() if v <= min_count}
        for u in self.utterances.values():
            text = u.text.split()
            new_text = []
            for t in text:
                if dictionary is not None:
                    dictionary.to_int(t)
                    lookup = dictionary.split_clitics(t)
                    if lookup is None:
                        continue
                else:
                    lookup = [t]
                for item in lookup:
                    if item in unk_words:
                        new_text.append('<unk>')
                    elif dictionary is not None and item not in dictionary.words:
                        new_text.append('<unk>')
                    else:
                        new_text.append(item)
            yield ' '.join(new_text)

    def subset_directory(self, subset:Optional[int]) -> str:
        if subset is None or subset > self.num_utterances or subset <= 0:
            for j in self.jobs:
                j.set_subset(None)
            return self.split_directory
        directory = os.path.join(self.output_directory, f'subset_{subset}')
        self.create_subset(subset)
        return directory

    def initialize_corpus(self, dictionary: Optional[DictionaryType]=None, feature_config: Optional[FeatureConfig]=None) -> None:
        if not self.files:
            raise CorpusError('There were no wav files found for transcribing this corpus. Please validate the corpus. '
                              'This error can also be caused if you\'re trying to find non-wav files without sox available '
                          'on the system path.')

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

    def initialize_utt_fsts(self) -> None:
        for j in self.jobs:
            j.output_utt_fsts(self)

    def create_subset(self, subset: Optional[int]) -> None:
        subset_directory = os.path.join(self.output_directory, f'subset_{subset}')

        larger_subset_num = subset * 10
        if larger_subset_num < self.num_utterances:
            # Get all shorter utterances that are not one word long
            utts = sorted((utt for utt in self.utterances.values() if ' ' in utt.text),
                          key=lambda x: x.duration)
            larger_subset = utts[:larger_subset_num]
        else:
            larger_subset = self.utterances.values()
        random.seed(1234)  # make it deterministic sampling
        subset_utts = set(random.sample(larger_subset, subset))
        log_dir = os.path.join(subset_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)

        for j in self.jobs:
            j.set_subset(subset_utts)
            j.output_to_directory(subset_directory)
