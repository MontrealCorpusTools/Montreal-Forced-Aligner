from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable, Collection
if TYPE_CHECKING:
    from ..corpus import AlignableCorpus
    from ..dictionary import Dictionary
    from ..config import AlignConfig
    from ..models import AcousticModel
    from logging import Logger

import os
import re
from collections import Counter

from .base import BaseAligner
from ..multiprocessing import generate_pronunciations


def parse_transitions(path, phones_path):
    state_extract_pattern = re.compile(r'Transition-state (\d+): phone = (\w+)')
    id_extract_pattern = re.compile(r'Transition-id = (\d+)')
    cur_phone = None
    current = 0
    with open(path, encoding='utf8') as f, open(phones_path, 'w', encoding='utf8') as outf:
        outf.write('{} {}\n'.format('<eps>', 0))
        for line in f:
            line = line.strip()
            if line.startswith('Transition-state'):
                m = state_extract_pattern.match(line)
                _, phone = m.groups()
                if phone != cur_phone:
                    current = 0
                    cur_phone = phone
            else:
                m = id_extract_pattern.match(line)
                transition_id = m.groups()[0]
                outf.write('{}_{} {}\n'.format(phone, current, transition_id))
                current += 1


class PretrainedAligner(BaseAligner):
    """
    Class for aligning a dataset using a pretrained acoustic model

    Parameters
    ----------
    corpus : :class:`~montreal_forced_aligner.corpus.AlignableCorpus`
        Corpus object for the dataset
    dictionary : :class:`~montreal_forced_aligner.dictionary.Dictionary`
        Dictionary object for the pronunciation dictionary
    acoustic_model : :class:`~montreal_forced_aligner.models.AcousticModel`
        Archive containing the acoustic model and pronunciation dictionary
    align_config : :class:`~montreal_forced_aligner.config.AlignConfig`
        Configuration for alignment
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``
    call_back : callable, optional
        Specifies a call back function for alignment
    """

    def __init__(self, corpus: AlignableCorpus, dictionary: Dictionary, acoustic_model: AcousticModel, align_config: AlignConfig,
                 temp_directory: Optional[str]=None,
                 call_back: Optional[Callable]=None, debug: bool=False, verbose: bool=False, logger: Optional[Logger]=None):
        self.acoustic_model = acoustic_model
        super().__init__(corpus, dictionary, align_config, temp_directory,
                                                call_back, debug, verbose, logger)
        self.data_directory = corpus.split_directory
        log_dir = os.path.join(self.align_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)
        self.align_config.logger = self.logger
        self.logger.info('Done with setup!')

    @property
    def model_directory(self) -> str:
        return os.path.join(self.temp_directory, 'model')

    def setup(self) -> None:
        self.dictionary.nonsil_phones = self.acoustic_model.meta['phones']
        super(PretrainedAligner, self).setup()
        self.acoustic_model.export_model(self.align_directory)

    @property
    def ali_paths(self):
        jobs = [x.align_arguments(self) for x in self.corpus.jobs]
        ali_paths = []
        for j in jobs:
            ali_paths.extend(j.ali_paths.values())
        return ali_paths

    def generate_pronunciations(self, output_path: str, calculate_silence_probs: bool=False, min_count: int=1) -> None:
        pron_counts, utt_mapping = generate_pronunciations(self)
        if self.dictionary.has_multiple:
            dictionary_mapping = self.dictionary.dictionary_mapping()
        else:
            dictionary_mapping = {self.dictionary.name: self.dictionary}
        for dict_name, dictionary in dictionary_mapping.items():
            counts = pron_counts[dict_name]
            mapping = utt_mapping[dict_name]
            if calculate_silence_probs:
                sil_before_counts = Counter()
                nonsil_before_counts = Counter()
                sil_after_counts = Counter()
                nonsil_after_counts = Counter()
                sils = ['<s>', '</s>', '<eps>']
                for u, v in mapping.items():
                    for i, w in enumerate(v):
                        if w in sils:
                            continue
                        prev_w = v[i - 1]
                        next_w = v[i + 1]
                        if prev_w in sils:
                            sil_before_counts[w] += 1
                        else:
                            nonsil_before_counts[w] += 1
                        if next_w in sils:
                            sil_after_counts[w] += 1
                        else:
                            nonsil_after_counts[w] += 1

            dictionary.pronunciation_probabilities = True
            for word, prons in dictionary.words.items():
                if word not in counts:
                    for p in prons:
                        p['probability'] = 1
                else:
                    total = 0
                    best_pron = 0
                    best_count = 0
                    for p in prons:
                        p['probability'] = min_count
                        if p['pronunciation'] in counts[word]:
                            p['probability'] += counts[word][p['pronunciation']]
                        total += p['probability']
                        if p['probability'] > best_count:
                            best_pron = p['pronunciation']
                            best_count = p['probability']
                    for p in prons:
                        if p['pronunciation'] == best_pron:
                            p['probability'] = 1
                        else:
                            p['probability'] /= total
                    dictionary.words[word] = prons
            dictionary.export_lexicon(output_path, probability=True)