import os
import shutil
from tqdm import tqdm
import time
import re

from .base import BaseAligner, TEMP_DIR, TriphoneFmllrConfig, TriphoneConfig

from ..dictionary import Dictionary
from ..corpus import load_scp,save_scp

from ..multiprocessing import (align, calc_fmllr, test_utterances,thirdparty_binary, subprocess,convert_ali_to_textgrids)


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
    archive : :class:`~aligner.archive.Archive`
        Archive containing the acoustic model and pronunciation dictionary
    corpus : :class:`~aligner.corpus.Corpus`
        Corpus object for the dataset
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
    def __init__(self, archive, corpus, output_directory,
                    temp_directory = None, num_jobs = 3, speaker_independent = False,
                    call_back = None, debug = False):
        self.debug = debug
        if temp_directory is None:
            temp_directory = TEMP_DIR
        self.temp_directory = temp_directory
        self.output_directory = output_directory
        self.corpus = corpus
        self.speaker_independent = speaker_independent
        self.dictionary = Dictionary(archive.dictionary_path, os.path.join(temp_directory, 'dictionary'),
                                     word_set=corpus.word_set, debug=debug)

        self.dictionary.write()
        archive.export_triphone_model(self.tri_directory)
        log_dir = os.path.join(self.tri_directory, 'log')
        os.makedirs(log_dir, exist_ok = True)

        if self.corpus.num_jobs != num_jobs:
            num_jobs = self.corpus.num_jobs
        self.num_jobs = num_jobs
        self.call_back = call_back
        if self.call_back is None:
            self.call_back = print
        self.verbose = False
        self.tri_fmllr_config = TriphoneFmllrConfig(**{'realign_iters': [1, 2],
                                                        'fmllr_iters': [1],
                                                        'num_iters': 3,
                                                       #'boost_silence': 0
                                                       })
        self.tri_config = TriphoneConfig()
        if self.debug:
            mdl_path = os.path.join(self.tri_directory,'final.mdl')
            tree_path = os.path.join(self.tri_directory,'tree')
            occs_path = os.path.join(self.tri_directory,'final.occs')
            log_path = os.path.join(self.tri_directory, 'log', 'show_transition.log')
            transition_path = os.path.join(self.tri_directory, 'transitions.txt')
            tree_pdf_path = os.path.join(self.tri_directory, 'tree.pdf')
            tree_dot_path = os.path.join(self.tri_directory, 'tree.dot')
            phones_path = os.path.join(self.dictionary.output_directory, 'phones.txt')
            triphones_path = os.path.join(self.tri_directory, 'triphones.txt')
            with open(log_path, 'w') as logf:
                with open(transition_path, 'w', encoding='utf8') as f:
                    subprocess.call([thirdparty_binary('show-transitions'), phones_path, mdl_path, occs_path], stdout=f, stderr=logf)
                parse_transitions(transition_path, triphones_path)
                if False:
                    with open(tree_dot_path, 'wb') as treef:
                        draw_tree_proc = subprocess.Popen([thirdparty_binary('draw-tree'), phones_path, tree_path], stdout=treef, stderr=logf)
                        draw_tree_proc.communicate()
                    with open(tree_dot_path, 'rb') as treeinf, open(tree_pdf_path, 'wb') as treef:
                        dot_proc = subprocess.Popen([thirdparty_binary('dot'), '-Tpdf', '-Gsize=8,10.5'], stdin=treeinf, stdout=treef, stderr=logf)
                        dot_proc.communicate()
        print('Done with setup.')

    def test_utterance_transcriptions(self):
        self.corpus.setup_splits(self.dictionary)
        return test_utterances(self)

    def do_align(self):
        '''
        Perform alignment while calculating speaker transforms (fMLLR estimation)
        '''
        self._init_tri()
        if not self.speaker_independent:
            self.train_tri_fmllr()

    def _align_fmllr(self):
        '''
        Align the dataset using speaker-adapted transforms
        '''
        model_directory = self.tri_directory
        output_directory = self.tri_ali_directory
        os.makedirs(output_directory, exist_ok=True)
        if self.debug:
            shutil.copyfile(os.path.join(self.tri_directory,'triphones.txt'),
                        os.path.join(self.tri_ali_directory,'triphones.txt'))
        self._align_si(fmllr = False)
        sil_phones = self.dictionary.silence_csl

        log_dir = os.path.join(output_directory, 'log')
        os.makedirs(log_dir, exist_ok = True)
        if not self.speaker_independent:
            calc_fmllr(output_directory, self.corpus.split_directory,
                        sil_phones, self.num_jobs, self.tri_fmllr_config, initial = True)
            optional_silence = self.dictionary.optional_silence_csl
            align(0, output_directory, self.corpus.split_directory,
                        optional_silence, self.num_jobs, self.tri_fmllr_config)

    def _init_tri(self):
        if not os.path.exists(self.tri_ali_directory):
            self._align_fmllr()
        if self.speaker_independent:
           return
        os.makedirs(os.path.join(self.tri_fmllr_directory, 'log'), exist_ok = True)
        begin = time.time()
        self.corpus.setup_splits(self.dictionary)

        shutil.copy(os.path.join(self.tri_directory,'final.mdl'),
                        os.path.join(self.tri_fmllr_directory,'1.mdl'))

        for i in range(self.num_jobs):
            shutil.copy(os.path.join(self.tri_ali_directory, 'fsts.{}'.format(i)),
                        os.path.join(self.tri_fmllr_directory, 'fsts.{}'.format(i)))
            shutil.copy(os.path.join(self.tri_ali_directory, 'trans.{}'.format(i)),
                        os.path.join(self.tri_fmllr_directory, 'trans.{}'.format(i)))

    def train_tri_fmllr(self):
        directory = self.tri_fmllr_directory
        sil_phones = self.dictionary.silence_csl
        if self.call_back == print:
            iters = tqdm(range(1, self.tri_fmllr_config.num_iters))
        else:
            iters = range(1, self.tri_fmllr_config.num_iters)
        log_directory = os.path.join(directory, 'log')
        for i in iters:
            model_path = os.path.join(directory,'{}.mdl'.format(i))
            occs_path = os.path.join(directory, '{}.occs'.format(i+1))
            next_model_path = os.path.join(directory,'{}.mdl'.format(i+1))
            if os.path.exists(next_model_path):
                continue
            align(i, directory, self.corpus.split_directory,
                            self.dictionary.optional_silence_csl,
                            self.num_jobs, self.tri_fmllr_config)
            calc_fmllr(directory, self.corpus.split_directory, sil_phones,
                        self.num_jobs, self.tri_fmllr_config, initial = False, iteration = i)
            os.rename(model_path, next_model_path)
            self.parse_log_directory(log_directory, i)
        os.rename(next_model_path, os.path.join(directory,'final.mdl'))

    def export_textgrids(self):
        '''
        Export a TextGrid file for every sound file in the dataset
        '''
        if self.speaker_independent:
            model_directory = self.tri_ali_directory
        else:
            model_directory = self.tri_fmllr_directory
        convert_ali_to_textgrids(self.output_directory, model_directory, self.dictionary,
                            self.corpus, self.num_jobs)
