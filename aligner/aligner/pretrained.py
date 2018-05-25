import os
import shutil
from tqdm import tqdm
import re
import glob

from .base import BaseAligner, TEMP_DIR, TriphoneFmllrConfig, TriphoneConfig, LdaMlltConfig, iVectorExtractorConfig, NnetBasicConfig

from ..exceptions import PronunciationAcousticMismatchError

from ..multiprocessing import (align, calc_fmllr, test_utterances, thirdparty_binary, subprocess,
                               convert_ali_to_textgrids, compile_train_graphs, nnet_get_align_feats, nnet_align)


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

    def __init__(self, corpus, dictionary, acoustic_model, output_directory,
                 temp_directory=None, num_jobs=3, speaker_independent=False,
                 call_back=None, debug=False, skip_input=False, nnet=False):
        self.debug = debug
        self.nnet = nnet
        if temp_directory is None:
            temp_directory = TEMP_DIR
        self.acoustic_model = acoustic_model
        self.temp_directory = temp_directory
        self.output_directory = output_directory
        self.corpus = corpus
        self.speaker_independent = speaker_independent
        self.dictionary = dictionary
        self.skip_input = skip_input
        self.setup()

        if not nnet:
            self.acoustic_model.export_triphone_model(self.tri_directory)
            log_dir = os.path.join(self.tri_directory, 'log')
        else:
            self.acoustic_model.export_nnet_model(self.nnet_basic_directory)
            log_dir = os.path.join(self.nnet_basic_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)

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
                                                       # 'boost_silence': 0
                                                       })
        self.tri_config = TriphoneConfig()
        self.lda_mllt_config = LdaMlltConfig()
        self.ivector_extractor_config = iVectorExtractorConfig()
        self.nnet_basic_config = NnetBasicConfig()

        if self.debug:
            os.makedirs(os.path.join(self.tri_directory, 'log'), exist_ok=True)
            mdl_path = os.path.join(self.tri_directory, 'final.mdl')
            tree_path = os.path.join(self.tri_directory, 'tree')
            occs_path = os.path.join(self.tri_directory, 'final.occs')
            log_path = os.path.join(self.tri_directory, 'log', 'show_transition.log')
            transition_path = os.path.join(self.tri_directory, 'transitions.txt')
            tree_pdf_path = os.path.join(self.tri_directory, 'tree.pdf')
            tree_dot_path = os.path.join(self.tri_directory, 'tree.dot')
            phones_path = os.path.join(self.dictionary.output_directory, 'phones.txt')
            triphones_path = os.path.join(self.tri_directory, 'triphones.txt')
            with open(log_path, 'w') as logf:
                with open(transition_path, 'w', encoding='utf8') as f:
                    subprocess.call([thirdparty_binary('show-transitions'), phones_path, mdl_path, occs_path], stdout=f,
                                    stderr=logf)
                parse_transitions(transition_path, triphones_path)
                if False:
                    with open(tree_dot_path, 'wb') as treef:
                        draw_tree_proc = subprocess.Popen([thirdparty_binary('draw-tree'), phones_path, tree_path],
                                                          stdout=treef, stderr=logf)
                        draw_tree_proc.communicate()
                    with open(tree_dot_path, 'rb') as treeinf, open(tree_pdf_path, 'wb') as treef:
                        dot_proc = subprocess.Popen([thirdparty_binary('dot'), '-Tpdf', '-Gsize=8,10.5'], stdin=treeinf,
                                                    stdout=treef, stderr=logf)
                        dot_proc.communicate()
        print('Done with setup.')

    def setup(self):
        self.dictionary.nonsil_phones = self.acoustic_model.meta['phones']
        super(PretrainedAligner, self).setup()
    def test_utterance_transcriptions(self):
        return test_utterances(self)

    def do_align_nnet(self):
        '''
        Perform alignment using a previous DNN model
        '''

       # N.B.: This if ought to be commented out when developing.
        #if not os.path.exists(self.nnet_basic_ali_directory):
        print("doing align nnet")
        print("nnet basic directory is: {}".format(self.nnet_basic_directory))
        optional_silence = self.dictionary.optional_silence_csl

        # Extract i-vectors
        self._extract_ivectors()

        # Compile train graphs (gets fsts.{} for alignment)
        compile_train_graphs(self.nnet_basic_directory, self.dictionary.output_directory,
                             self.corpus.split_directory, self.num_jobs, mdl='final')

        # Get alignment feats
        nnet_get_align_feats(self.nnet_basic_directory, self.corpus.split_directory, self.extracted_ivector_directory, self.nnet_basic_config, self.num_jobs)

        # Do nnet alignment
        nnet_align(0, self.nnet_basic_directory, optional_silence, self.num_jobs, self.nnet_basic_config, mdl='final')

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
        model_directory = self.tri_directory        # Get final.mdl from here
        first_output_directory = self.tri_ali_directory
        second_output_directory = self.tri_fmllr_ali_directory
        os.makedirs(first_output_directory, exist_ok=True)
        os.makedirs(second_output_directory, exist_ok=True)
        if self.debug:
            shutil.copyfile(os.path.join(self.tri_directory, 'triphones.txt'),
                            os.path.join(self.tri_ali_directory, 'triphones.txt'))
        self._align_si(fmllr=False)
        sil_phones = self.dictionary.silence_csl

        log_dir = os.path.join(first_output_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)
        if not self.speaker_independent:
            calc_fmllr(first_output_directory, self.corpus.split_directory,
                       sil_phones, self.num_jobs, self.tri_fmllr_config, initial=True)
            optional_silence = self.dictionary.optional_silence_csl
            align(0, first_output_directory, self.corpus.split_directory,
                  optional_silence, self.num_jobs, self.tri_fmllr_config)

        # Copy into the "correct" tri_fmllr_ali output directory
        for file in glob.glob(os.path.join(first_output_directory, 'ali.*')):
            shutil.copy(file, second_output_directory)
        shutil.copy(os.path.join(first_output_directory, 'tree'), second_output_directory)
        shutil.copy(os.path.join(first_output_directory, 'final.mdl'), second_output_directory)

    def _init_tri(self):
        if not os.path.exists(self.tri_ali_directory):
            self._align_fmllr()
        if self.speaker_independent:
            return
        os.makedirs(os.path.join(self.tri_fmllr_directory, 'log'), exist_ok=True)

        shutil.copy(os.path.join(self.tri_directory, 'final.mdl'),
                    os.path.join(self.tri_fmllr_directory, '1.mdl'))

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
            model_path = os.path.join(directory, '{}.mdl'.format(i))
            occs_path = os.path.join(directory, '{}.occs'.format(i + 1))
            next_model_path = os.path.join(directory, '{}.mdl'.format(i + 1))
            if os.path.exists(next_model_path):
                continue
            align(i, directory, self.corpus.split_directory,
                  self.dictionary.optional_silence_csl,
                  self.num_jobs, self.tri_fmllr_config)
            calc_fmllr(directory, self.corpus.split_directory, sil_phones,
                       self.num_jobs, self.tri_fmllr_config, initial=False, iteration=i)
            os.rename(model_path, next_model_path)
            self.parse_log_directory(log_directory, i)
        os.rename(next_model_path, os.path.join(directory, 'final.mdl'))

    def export_textgrids(self):
        '''
        Export a TextGrid file for every sound file in the dataset
        '''
        if self.speaker_independent:
            model_directory = self.tri_ali_directory
        else:
            model_directory = self.tri_fmllr_directory
        if self.nnet:
            model_directory = self.nnet_basic_directory
        convert_ali_to_textgrids(self.output_directory, model_directory, self.dictionary,
                                 self.corpus, self.num_jobs)
        print("Exported textgrids to {}".format(self.output_directory))
        print("Log of export at {}".format(model_directory))
