import os
import shutil
from tqdm import tqdm
import re

from .base import BaseAligner

from ..exceptions import PronunciationAcousticMismatchError

from ..multiprocessing import (align, calc_fmllr, test_utterances, thirdparty_binary, subprocess,
                               convert_ali_to_textgrids)


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
                id = m.groups()[0]
                outf.write('{}_{} {}\n'.format(phone, current, id))
                current += 1


class PretrainedAligner(BaseAligner):
    '''
    Class for aligning a dataset using a pretrained acoustic model

    Parameters
    ----------
    corpus : :class:`~aligner.corpus.Corpus`
        Corpus object for the dataset
    dictionary : :class:`~aligner.dictionary.Dictionary`
        Dictionary object for the pronunciation dictionary
    acoustic_model : :class:`~aligner.models.AcousticModel`
        Archive containing the acoustic model and pronunciation dictionary
    align_config : :class:`~aligner.config.AlignConfig`
        Configuration for alignment
    output_directory : str
        Path to directory to save TextGrids
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``
    num_jobs : int, optional
        Number of processes to use, defaults to 3
    call_back : callable, optional
        Specifies a call back function for alignment
    '''

    def __init__(self, corpus, dictionary, acoustic_model, align_config, output_directory,
                 temp_directory=None, num_jobs=3,
                 call_back=None, debug=False, skip_input=False, verbose=False):
        self.acoustic_model = acoustic_model
        super(PretrainedAligner, self).__init__(corpus, dictionary, align_config, output_directory, temp_directory, num_jobs,
                 call_back, debug, skip_input, verbose)

        self.acoustic_model.export_model(self.model_directory)
        log_dir = os.path.join(self.model_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)

        print('Done with setup.')

    @property
    def model_directory(self):
        return os.path.join(self.temp_directory, 'model')

    @property
    def align_directory(self):
        return os.path.join(self.temp_directory, 'align')

    def setup(self):
        self.dictionary.nonsil_phones = self.acoustic_model.meta['phones']
        super(PretrainedAligner, self).setup()

    def test_utterance_transcriptions(self):
        return test_utterances(self)

    def align(self, call_back=None):
        data_directory = self.corpus.split_directory()
        log_dir = os.path.join(self.align_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)
        align('final', self.model_directory, data_directory,
              self.dictionary.optional_silence_csl,
              self.corpus.num_jobs, self.align_config, self.align_directory)

    def export_textgrids(self):
        '''
        Export a TextGrid file for every sound file in the dataset
        '''
        ali_directory = self.align_directory
        convert_ali_to_textgrids(self.align_config, self.output_directory, ali_directory, self.dictionary,
                                 self.corpus, self.num_jobs)
        self.compile_information(ali_directory)