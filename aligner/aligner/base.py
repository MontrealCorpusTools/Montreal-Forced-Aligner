import os
import shutil
import glob
import subprocess
import re
import io
import math
from tqdm import tqdm
from shutil import copy, copyfile, rmtree, make_archive, unpack_archive
from contextlib import redirect_stdout
from aligner.models import IvectorExtractor
from random import shuffle

from ..helper import thirdparty_binary, make_path_safe, awk_like, filter_scp

from ..config import (MonophoneConfig, TriphoneConfig, TriphoneFmllrConfig,
                      LdaMlltConfig, DiagUbmConfig, iVectorExtractorConfig,
                      NnetBasicConfig)

from ..multiprocessing import (align, mono_align_equal, compile_train_graphs,
                               acc_stats, tree_stats, convert_alignments,
                               convert_ali_to_textgrids, calc_fmllr,
                               lda_acc_stats,
                               calc_lda_mllt, gmm_gselect, acc_global_stats,
                               gauss_to_post, acc_ivector_stats, get_egs,
                               get_lda_nnet, nnet_train_trans, nnet_train,
                               nnet_align, nnet_get_align_feats, extract_ivectors,
                               compute_prob, get_average_posteriors, relabel_egs)
#from ..accuracy_graph import get_accuracy_graph


from ..exceptions import NoSuccessfulAlignments

from .. import __version__

from ..config import TEMP_DIR


class BaseAligner(object):
    '''
    Base aligner class for common aligner functions

    Parameters
    ----------
    corpus : :class:`~aligner.corpus.Corpus`
        Corpus object for the dataset
    dictionary : :class:`~aligner.dictionary.Dictionary`
        Dictionary object for the pronunciation dictionary
    output_directory : str
        Path to export aligned TextGrids
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``
    num_jobs : int, optional
        Number of processes to use, defaults to 3
    call_back : callable, optional
        Specifies a call back function for alignment
    mono_params : :class:`~aligner.config.MonophoneConfig`, optional
        Monophone training parameters to use, if different from defaults
    tri_params : :class:`~aligner.config.TriphoneConfig`, optional
        Triphone training parameters to use, if different from defaults
    tri_fmllr_params : :class:`~aligner.config.TriphoneFmllrConfig`, optional
        Speaker-adapted triphone training parameters to use, if different from defaults
    '''

    def __init__(self, corpus, dictionary, output_directory,
                 temp_directory=None, num_jobs=3, call_back=None,
                 mono_params=None, tri_params=None,
                 tri_fmllr_params=None, lda_mllt_params=None,
                 diag_ubm_params=None, ivector_extractor_params=None,
                 nnet_basic_params=None,
                 debug=False, skip_input=False, nnet=False):
        self.nnet = nnet

        if mono_params is None:
            mono_params = {}
        if tri_params is None:
            tri_params = {}
        if tri_fmllr_params is None:
            tri_fmllr_params = {}

        if lda_mllt_params is None:
            lda_mllt_params = {}
        if diag_ubm_params is None:
            diag_ubm_params = {}
        if ivector_extractor_params is None:
            ivector_extractor_params = {}
        if nnet_basic_params is None:
            nnet_basic_params = {}

        self.mono_config = MonophoneConfig(**mono_params)
        self.tri_config = TriphoneConfig(**tri_params)
        self.tri_fmllr_config = TriphoneFmllrConfig(**tri_fmllr_params)

        self.lda_mllt_config = LdaMlltConfig(**lda_mllt_params)
        self.diag_ubm_config = DiagUbmConfig(**diag_ubm_params)
        self.ivector_extractor_config = iVectorExtractorConfig(**ivector_extractor_params)
        self.nnet_basic_config = NnetBasicConfig(**nnet_basic_params)

        self.corpus = corpus
        self.dictionary = dictionary
        self.output_directory = output_directory
        self.num_jobs = num_jobs
        if self.corpus.num_jobs != num_jobs:
            self.num_jobs = self.corpus.num_jobs
        if not temp_directory:
            temp_directory = TEMP_DIR
        self.temp_directory = temp_directory
        self.call_back = call_back
        if self.call_back is None:
            self.call_back = print
        self.verbose = False
        self.debug = debug
        self.skip_input = skip_input
        self.setup()

    def setup(self):
        self.dictionary.write()
        self.corpus.initialize_corpus(self.dictionary, skip_input=self.skip_input)
        print(self.corpus.speaker_utterance_info())

    @property
    def meta(self):
        data = {'phones':sorted(self.dictionary.nonsil_phones),
                'version': __version__,
                'architecture':'gmm-hmm',
                'features':'mfcc+deltas',
                }
        return data

    @property
    def mono_directory(self):
        return os.path.join(self.temp_directory, 'mono')

    @property
    def mono_final_model_path(self):
        return os.path.join(self.mono_directory, 'final.mdl')

    @property
    def mono_ali_directory(self):
        return os.path.join(self.temp_directory, 'mono_ali')

    @property
    def tri_directory(self):
        return os.path.join(self.temp_directory, 'tri')

    @property
    def tri_ali_directory(self):
        return os.path.join(self.temp_directory, 'tri_ali')

    @property
    def tri_final_model_path(self):
        return os.path.join(self.tri_directory, 'final.mdl')

    @property
    def tri_fmllr_directory(self):
        return os.path.join(self.temp_directory, 'tri_fmllr')

    @property
    def tri_fmllr_ali_directory(self):
        return os.path.join(self.temp_directory, 'tri_fmllr_ali')

    @property
    def tri_fmllr_final_model_path(self):
        return os.path.join(self.tri_fmllr_directory, 'final.mdl')

    # Beginning of nnet properties
    @property
    def lda_mllt_directory(self):
        return os.path.join(self.temp_directory, 'lda_mllt')

    @property
    def lda_mllt_ali_directory(self):
        return os.path.join(self.temp_directory, 'lda_mllt_ali')

    @property
    def lda_mllt_final_model_path(self):
        return os.path.join(self.lda_mllt_directory, 'final.mdl')

    @property
    def diag_ubm_directory(self):
        return os.path.join(self.temp_directory, 'diag_ubm')

    @property
    def diag_ubm_final_model_path(self):
        return os.path.join(self.diag_ubm_directory, 'final.dubm')

    @property
    def ivector_extractor_directory(self):
        return os.path.join(self.temp_directory, 'ivector_extractor')

    @property
    def ivector_extractor_final_model_path(self):
        return os.path.join(self.ivector_extractor_directory, 'final.ie')

    @property
    def extracted_ivector_directory(self):
        return os.path.join(self.temp_directory, 'extracted_ivector')

    @property
    def nnet_basic_directory(self):
        return os.path.join(self.temp_directory, 'nnet_basic')

    @property
    def nnet_basic_ali_directory(self):
        return os.path.join(self.temp_directory, 'nnet_basic_ali')

    @property
    def nnet_basic_final_model_path(self):
        return os.path.join(self.nnet_basic_directory, 'final.mdl')

    # End of nnet properties

    def export_textgrids(self):
        '''
        Export a TextGrid file for every sound file in the dataset
        '''
        if os.path.exists(self.nnet_basic_final_model_path):
            model_directory = self.nnet_basic_directory
        elif os.path.exists(self.tri_fmllr_final_model_path):
            model_directory = self.tri_fmllr_directory
        elif os.path.exists(self.tri_final_model_path):
            model_directory = self.tri_directory
        elif os.path.exists(self.mono_final_model_path):
            model_directory = self.mono_directory

        convert_ali_to_textgrids(self.output_directory, model_directory, self.dictionary,
                                 self.corpus, self.num_jobs)

    def get_num_gauss_mono(self):
        '''
        Get the number of gaussians for a monophone model
        '''
        with open(os.devnull, 'w') as devnull:
            proc = subprocess.Popen([thirdparty_binary('gmm-info'),
                                     '--print-args=false',
                                     os.path.join(self.mono_directory, '0.mdl')],
                                    stderr=devnull,
                                    stdout=subprocess.PIPE)
            stdout, stderr = proc.communicate()
            num = stdout.decode('utf8')
            matches = re.search(r'gaussians (\d+)', num)
            num = int(matches.groups()[0])
        return num

    def _align_si(self, fmllr=False, lda_mllt=False, feature_name=None):
        '''
        Generate an alignment of the dataset
        '''
        if fmllr and os.path.exists(self.tri_fmllr_final_model_path):
            model_directory = self.tri_fmllr_directory
            output_directory = self.tri_fmllr_ali_directory
            config = self.tri_fmllr_config

        elif fmllr:     # First pass with fmllr, final path doesn't exist yet
            model_directory = self.tri_directory
            output_directory = self.tri_fmllr_ali_directory
            config = self.tri_fmllr_config

        elif lda_mllt and os.path.exists(self.lda_mllt_final_model_path):
            model_directory = self.lda_mllt_directory
            output_directory = self.lda_mllt_ali_directory
            config = self.lda_mllt_config

        elif lda_mllt:  # First pass with LDA + MLLT, final path doesn't exist yet
            model_directory = self.tri_fmllr_directory
            output_directory = self.lda_mllt_ali_directory
            config = self.lda_mllt_config
        elif os.path.exists(self.tri_final_model_path):
            model_directory = self.tri_directory
            output_directory = self.tri_ali_directory
            config = self.tri_config
        elif os.path.exists(self.mono_final_model_path):
            model_directory = self.mono_directory
            output_directory = self.mono_ali_directory
            config = self.mono_config

        optional_silence = self.dictionary.optional_silence_csl
        oov = self.dictionary.oov_int

        log_dir = os.path.join(output_directory, 'log')
        os.makedirs(self.tri_fmllr_ali_directory, exist_ok=True)
        os.makedirs(self.lda_mllt_ali_directory, exist_ok=True)

        os.makedirs(log_dir, exist_ok=True)

        shutil.copyfile(os.path.join(model_directory, 'tree'),
                        os.path.join(output_directory, 'tree'))
        shutil.copyfile(os.path.join(model_directory, 'final.mdl'),
                        os.path.join(output_directory, '0.mdl'))

        shutil.copyfile(os.path.join(model_directory, 'final.occs'),
                        os.path.join(output_directory, '0.occs'))

        feat_type = 'delta'

        compile_train_graphs(output_directory, self.dictionary.output_directory,
                             self.corpus.split_directory, self.num_jobs, debug=self.debug)

        align(0, output_directory, self.corpus.split_directory,
              optional_silence, self.num_jobs, config, feature_name=feature_name)
        shutil.copyfile(os.path.join(output_directory, '0.mdl'), os.path.join(output_directory, 'final.mdl'))
        shutil.copyfile(os.path.join(output_directory, '0.occs'), os.path.join(output_directory, 'final.occs'))

        if output_directory == self.tri_fmllr_ali_directory:
            os.makedirs(self.tri_fmllr_directory, exist_ok=True)
            shutil.copyfile(os.path.join(output_directory, '0.mdl'), os.path.join(self.tri_fmllr_directory, 'final.mdl'))
            shutil.copyfile(os.path.join(output_directory, '0.occs'), os.path.join(self.tri_fmllr_directory, 'final.occs'))
        elif output_directory == self.lda_mllt_ali_directory:
            os.makedirs(self.lda_mllt_directory, exist_ok=True)
            shutil.copyfile(os.path.join(output_directory, '0.mdl'), os.path.join(self.lda_mllt_directory, 'final.mdl'))
            shutil.copyfile(os.path.join(output_directory, '0.occs'), os.path.join(self.lda_mllt_directory, 'final.occs'))

    def parse_log_directory(self, directory, iteration):
        '''
        Parse error files and relate relevant information about unaligned files
        '''
        if not self.verbose:
            return
        error_regex = re.compile(r'Did not successfully decode file (\w+),')
        too_little_data_regex = re.compile(
            r'Gaussian has too little data but not removing it because it is the last Gaussian')
        skipped_transition_regex = re.compile(r'(\d+) out of (\d+) transition-states skipped due to insuffient data')

        log_like_regex = re.compile(r'Overall avg like per frame = ([-0-9.]+|nan) over (\d+) frames')
        error_files = []
        for i in range(self.num_jobs):
            path = os.path.join(directory, 'align.{}.{}.log'.format(iteration - 1, i))
            if not os.path.exists(path):
                continue
            with open(path, 'r') as f:
                error_files.extend(error_regex.findall(f.read()))
        update_path = os.path.join(directory, 'update.{}.log'.format(iteration))
        if os.path.exists(update_path):
            with open(update_path, 'r') as f:
                data = f.read()
                m = log_like_regex.search(data)
                if m is not None:
                    log_like, tot_frames = m.groups()
                    if log_like == 'nan':
                        raise (NoSuccessfulAlignments('Could not align any files.  Too little data?'))
                    self.call_back('log-likelihood', float(log_like))
                skipped_transitions = skipped_transition_regex.search(data)
                self.call_back('skipped transitions', *skipped_transitions.groups())
                num_too_little_data = len(too_little_data_regex.findall(data))
                self.call_back('missing data gaussians', num_too_little_data)
        if error_files:
            self.call_back('could not align', error_files)

    def _align_fmllr(self):
        '''
        Align the dataset using speaker-adapted transforms
        '''
        model_directory = self.tri_directory        # Get final.mdl from here
        first_output_directory = self.tri_ali_directory
        second_output_directory = self.tri_fmllr_ali_directory
        self._align_si(fmllr=False)
        sil_phones = self.dictionary.silence_csl

        log_dir = os.path.join(first_output_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)

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


    def _init_tri(self, fmllr=False):
        if fmllr:
            config = self.tri_fmllr_config
            directory = self.tri_fmllr_directory
            align_directory = self.tri_ali_directory
        else:
            config = self.tri_config
            directory = self.tri_directory
            align_directory = self.mono_ali_directory

        if not self.debug:
            if os.path.exists(os.path.join(directory, '1.mdl')):
                return

        if fmllr:
            print('Initializing speaker-adapted triphone training...')
        else:
            print('Initializing triphone training...')
        context_opts = []
        ci_phones = self.dictionary.silence_csl

        tree_stats(directory, align_directory,
                   self.corpus.split_directory, ci_phones, self.num_jobs)
        log_path = os.path.join(directory, 'log', 'questions.log')
        tree_path = os.path.join(directory, 'tree')
        treeacc_path = os.path.join(directory, 'treeacc')
        sets_int_path = os.path.join(self.dictionary.phones_dir, 'sets.int')
        roots_int_path = os.path.join(self.dictionary.phones_dir, 'roots.int')
        extra_question_int_path = os.path.join(self.dictionary.phones_dir, 'extra_questions.int')
        topo_path = os.path.join(self.dictionary.output_directory, 'topo')
        questions_path = os.path.join(directory, 'questions.int')
        questions_qst_path = os.path.join(directory, 'questions.qst')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('cluster-phones')] + context_opts +
                            [treeacc_path, sets_int_path, questions_path], stderr=logf)

        with open(extra_question_int_path, 'r') as inf, \
                open(questions_path, 'a') as outf:
            for line in inf:
                outf.write(line)

        log_path = os.path.join(directory, 'log', 'compile_questions.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('compile-questions')] + context_opts +
                            [topo_path, questions_path, questions_qst_path],
                            stderr=logf)

        log_path = os.path.join(directory, 'log', 'build_tree.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('build-tree')] + context_opts +
                            ['--verbose=1', '--max-leaves={}'.format(config.initial_gauss_count),
                             '--cluster-thresh={}'.format(config.cluster_threshold),
                             treeacc_path, roots_int_path, questions_qst_path,
                             topo_path, tree_path], stderr=logf)

        log_path = os.path.join(directory, 'log', 'init_model.log')
        occs_path = os.path.join(directory, '0.occs')
        mdl_path = os.path.join(directory, '0.mdl')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('gmm-init-model'),
                             '--write-occs=' + occs_path, tree_path, treeacc_path,
                             topo_path, mdl_path], stderr=logf)

        log_path = os.path.join(directory, 'log', 'mixup.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('gmm-mixup'),
                             '--mix-up={}'.format(config.initial_gauss_count),
                             mdl_path, occs_path, mdl_path], stderr=logf)

        #os.remove(treeacc_path)

        compile_train_graphs(directory, self.dictionary.output_directory,
                             self.corpus.split_directory, self.num_jobs)

        shutil.copy(occs_path, os.path.join(directory, '1.occs'))
        shutil.copy(mdl_path, os.path.join(directory, '1.mdl'))

        convert_alignments(directory, align_directory, self.num_jobs)

        if os.path.exists(os.path.join(align_directory, 'trans.0')):
            for i in range(self.num_jobs):
                shutil.copy(os.path.join(align_directory, 'trans.{}'.format(i)),
                            os.path.join(directory, 'trans.{}'.format(i)))



    def train_tri_fmllr(self):
        '''
        Perform speaker-adapted triphone training
        '''
        if not self.debug:
            if os.path.exists(self.tri_fmllr_final_model_path):
                print('Triphone FMLLR training already done, using previous final.mdl')
                return

        if not os.path.exists(self.tri_ali_directory):
            self._align_fmllr()

        #self._align_fmllr()

        os.makedirs(os.path.join(self.tri_fmllr_directory, 'log'), exist_ok=True)
        self._init_tri(fmllr=True)
        self._do_tri_fmllr_training()

    def _do_tri_fmllr_training(self):
        self.call_back('Beginning speaker-adapted triphone training...')
        self._do_training(self.tri_fmllr_directory, self.tri_fmllr_config)

    def _do_training(self, directory, config):
        if config.realign_iters is None:
            config.realign_iters = list(range(0, config.num_iters, 10))
        num_gauss = config.initial_gauss_count
        sil_phones = self.dictionary.silence_csl
        inc_gauss = config.inc_gauss_count
        if self.call_back == print:
            iters = tqdm(range(1, config.num_iters))
        else:
            iters = range(1, config.num_iters)
        log_directory = os.path.join(directory, 'log')
        for i in iters:
            model_path = os.path.join(directory, '{}.mdl'.format(i))
            occs_path = os.path.join(directory, '{}.occs'.format(i + 1))
            next_model_path = os.path.join(directory, '{}.mdl'.format(i + 1))

            if not self.debug:
                if os.path.exists(next_model_path):
                    continue

            if i in config.realign_iters:
                align(i, directory, self.corpus.split_directory,
                      self.dictionary.optional_silence_csl,
                      self.num_jobs, config,
                      feature_name='cmvnsplicetransformfeats')
            if config.do_fmllr and i in config.fmllr_iters:
                calc_fmllr(directory, self.corpus.split_directory, sil_phones,
                           self.num_jobs, config, initial=False, iteration=i)

            if config.do_lda_mllt and i <= config.num_iters:
                calc_lda_mllt(directory, self.corpus.split_directory,   # Could change this to make ali directory later
                #calc_lda_mllt(self.lda_mllt_ali_directory, sil_phones,
                              self.lda_mllt_directory, sil_phones,
                              self.num_jobs, config, config.num_iters,
                              initial=False, iteration=i, corpus=self.corpus)


            acc_stats(i, directory, self.corpus.split_directory, self.num_jobs,
                      config.do_fmllr, do_lda_mllt=config.do_lda_mllt)
            log_path = os.path.join(log_directory, 'update.{}.log'.format(i))
            with open(log_path, 'w') as logf:
                acc_files = [os.path.join(directory, '{}.{}.acc'.format(i, x))
                             for x in range(self.num_jobs)]
                est_proc = subprocess.Popen([thirdparty_binary('gmm-est'),
                                             '--write-occs=' + occs_path,
                                             '--mix-up=' + str(num_gauss), '--power=' + str(config.power),
                                             model_path,
                                             "{} - {}|".format(thirdparty_binary('gmm-sum-accs'),
                                                               ' '.join(map(make_path_safe, acc_files))),
                                             next_model_path],
                                            stderr=logf)
                est_proc.communicate()
            self.parse_log_directory(log_directory, i)
            if i < config.max_iter_inc:
                num_gauss += inc_gauss

        shutil.copy(os.path.join(directory, '{}.mdl'.format(config.num_iters)),
                    os.path.join(directory, 'final.mdl'))

        shutil.copy(os.path.join(directory, '{}.occs'.format(config.num_iters)),
                    os.path.join(directory, 'final.occs'))

        if config.do_lda_mllt:
            shutil.copy(os.path.join(directory, '{}.mat'.format(config.num_iters-1)),
                        os.path.join(directory, 'final.mat'))

    def train_lda_mllt(self):
        '''
        Perform LDA + MLLT training
        '''

        if not self.debug:
            if os.path.exists(self.lda_mllt_final_model_path):
                print('LDA + MLLT training already done, using previous final.mdl')
                return

        # N.B: The function _align_lda_mllt() is half-developed, but there doesn't seem to
        # be a reason for it to actually ever be called (since people will always have
        # fmllr done immediately before in the pipeline. Can clean/delete later if determined
        # that we need to actually use it somewhere or not).
        #if not os.path.exists(self.lda_mllt_ali_directory):
        #    self._align_lda_mllt()
        #self._align_lda_mllt()  # half implemented, can come back later or make people run from fmllr

        os.makedirs(os.path.join(self.lda_mllt_directory, 'log'), exist_ok=True)

        self._init_lda_mllt()
        self._do_lda_mllt_training()

    def _init_lda_mllt(self):
        '''
        Initialize LDA + MLLT training.
        '''
        config = self.lda_mllt_config
        directory = self.lda_mllt_directory
        align_directory = self.tri_fmllr_ali_directory  # The previous
        mdl_dir = self.tri_fmllr_directory

        if not self.debug:
            if os.path.exists(os.path.join(directory, '1.mdl')):
                return

        print('Initializing LDA + MLLT training...')

        context_opts = []
        ci_phones = self.dictionary.silence_csl

        log_path = os.path.join(directory, 'log', 'questions.log')
        tree_path = os.path.join(directory, 'tree')
        treeacc_path = os.path.join(directory, 'treeacc')
        sets_int_path = os.path.join(self.dictionary.phones_dir, 'sets.int')
        roots_int_path = os.path.join(self.dictionary.phones_dir, 'roots.int')
        extra_question_int_path = os.path.join(self.dictionary.phones_dir, 'extra_questions.int')
        topo_path = os.path.join(self.dictionary.output_directory, 'topo')
        questions_path = os.path.join(directory, 'questions.int')
        questions_qst_path = os.path.join(directory, 'questions.qst')

        final_mdl_path = os.path.join(self.tri_fmllr_directory)

        # Accumulate LDA stats
        lda_acc_stats(directory, self.corpus.split_directory, align_directory, config, ci_phones, self.num_jobs)

        # Accumulating tree stats
        self.corpus._norm_splice_transform_feats(self.lda_mllt_directory)
        tree_stats(directory, align_directory, self.corpus.split_directory, ci_phones,
                   self.num_jobs, feature_name='cmvnsplicetransformfeats')

        # Getting questions for tree clustering
        log_path = os.path.join(directory, 'log', 'cluster_phones.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('cluster-phones')] + context_opts +
                            [treeacc_path, sets_int_path, questions_path], stderr=logf)

        with open(extra_question_int_path, 'r') as inf, \
                open(questions_path, 'a') as outf:
            for line in inf:
                outf.write(line)

        log_path = os.path.join(directory, 'log', 'compile_questions.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('compile-questions')] + context_opts +
                            [topo_path, questions_path, questions_qst_path],
                            stderr=logf)

        # Building the tree
        log_path = os.path.join(directory, 'log', 'build_tree.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('build-tree')] + context_opts +
                            ['--verbose=1', '--max-leaves={}'.format(config.initial_gauss_count),
                             '--cluster-thresh={}'.format(config.cluster_threshold),
                             treeacc_path, roots_int_path, questions_qst_path,
                             topo_path, tree_path], stderr=logf)

        # Initializing the model
        log_path = os.path.join(directory, 'log', 'init_model.log')
        occs_path = os.path.join(directory, '0.occs')
        mdl_path = os.path.join(directory, '0.mdl')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('gmm-init-model'),
                             '--write-occs=' + occs_path, tree_path, treeacc_path,
                             topo_path, mdl_path], stderr=logf)
        shutil.copy(mdl_path, os.path.join(directory, '1.mdl'))
        shutil.copy(occs_path, os.path.join(directory, '1.occs'))

        convert_alignments(directory, align_directory, self.num_jobs)

        compile_train_graphs(directory, self.dictionary.output_directory,
                             self.corpus.split_directory, self.num_jobs)

        if os.path.exists(os.path.join(align_directory, 'trans.0')):            
            for i in range(self.num_jobs):                                      
                shutil.copy(os.path.join(align_directory, 'trans.{}'.format(i)),
                            os.path.join(directory, 'trans.{}'.format(i)))

    def _do_lda_mllt_training(self):
        self.call_back('Beginning LDA + MLLT training...')
        self._do_training(self.lda_mllt_directory, self.lda_mllt_config)

    def train_nnet_basic(self):
        '''
        Perform neural network training
        '''

        os.makedirs(os.path.join(self.nnet_basic_directory, 'log'), exist_ok=True)

        split_directory = self.corpus.split_directory
        config = self.nnet_basic_config
        tri_fmllr_config = self.tri_fmllr_config
        directory = self.nnet_basic_directory
        nnet_align_directory = self.nnet_basic_ali_directory
        align_directory = self.tri_fmllr_ali_directory
        lda_directory = self.lda_mllt_directory
        egs_directory = os.path.join(directory, 'egs')
        training_directory = self.corpus.output_directory

        sets_int_path = os.path.join(self.dictionary.phones_dir, 'sets.int')
        roots_int_path = os.path.join(self.dictionary.phones_dir, 'roots.int')
        extra_question_int_path = os.path.join(self.dictionary.phones_dir, 'extra_questions.int')
        topo_path = os.path.join(self.dictionary.output_directory, 'topo')
        L_fst_path = os.path.join(self.dictionary.output_directory, 'L.fst')
        ali_tree_path = os.path.join(align_directory, 'tree')
        shutil.copy(ali_tree_path, os.path.join(directory, 'tree'))

        mdl_path = os.path.join(align_directory, 'final.mdl')
        raw_feats = os.path.join(training_directory, 'feats.scp')

        tree_info_proc = subprocess.Popen([thirdparty_binary('tree-info'),
                                          os.path.join(align_directory, 'tree')],
                                          stdout=subprocess.PIPE)
        tree_info = tree_info_proc.stdout.read()
        tree_info = tree_info.split()
        num_leaves = tree_info[1]
        num_leaves = num_leaves.decode("utf-8")

        lda_dim = self.lda_mllt_config.dim 

        # Extract iVectors
        self._extract_ivectors()

        # Get LDA matrix
        fixed_ivector_dir = self.extracted_ivector_directory
        get_lda_nnet(directory, align_directory, fixed_ivector_dir, training_directory,
                     split_directory, raw_feats, self.dictionary.optional_silence_csl, config, self.num_jobs)

        log_path = os.path.join(directory, 'log', 'lda_matrix.log')
        with open(log_path, 'w') as logf:
            acc_files = [os.path.join(directory, 'lda.{}.acc'.format(x))
                         for x in range(self.num_jobs)]
            sum_lda_accs_proc = subprocess.Popen([thirdparty_binary('sum-lda-accs'),
                                                 os.path.join(directory, 'lda.acc')]
                                                 + acc_files,
                                                 stderr=logf)
            sum_lda_accs_proc.communicate()

            lda_mat_proc = subprocess.Popen([thirdparty_binary('nnet-get-feature-transform'),
                                            '--dim=' + str(lda_dim),
                                            os.path.join(directory, 'lda.mat'),
                                            os.path.join(directory, 'lda.acc')],
                                            stderr=logf)
            lda_mat_proc.communicate()
        lda_mat_path = os.path.join(directory, 'lda.mat')


        # Get examples for training
        os.makedirs(egs_directory, exist_ok=True)

        # # Get valid uttlist and train subset uttlist
        valid_uttlist = os.path.join(directory, 'valid_uttlist')
        train_subset_uttlist = os.path.join(directory, 'train_subset_uttlist')
        training_feats = os.path.join(directory, 'nnet_training_feats')
        num_utts_subset = 300
        log_path = os.path.join(directory, 'log', 'training_egs_feats.log')

        with open(log_path, 'w') as logf:
            with open(valid_uttlist, 'w') as outf:
                # Get first column from utt2spk (awk-like)
                utt2spk_col = awk_like(os.path.join(training_directory, 'utt2spk'), 0)
                # Shuffle the list from the column
                shuffle(utt2spk_col)
                # Take only the first num_utts_subset lines
                utt2spk_col = utt2spk_col[:num_utts_subset]
                # Write the result to file
                for line in utt2spk_col:
                    outf.write(line)
                    outf.write('\n')

            with open(train_subset_uttlist, 'w') as outf:
                # Get first column from utt2spk (awk-like)
                utt2spk_col = awk_like(os.path.join(training_directory, 'utt2spk'), 0)
                # Filter by the scp list
                filtered = filter_scp(valid_uttlist, utt2spk_col, exclude=True)
                # Shuffle the list
                shuffle(filtered)
                # Take only the first num_utts_subset lines
                filtered = filtered[:num_utts_subset]
                # Write the result to a file
                for line in filtered:
                    outf.write(line)
                    outf.write('\n')

        get_egs(directory, egs_directory, training_directory, split_directory, align_directory,
                fixed_ivector_dir, training_feats, valid_uttlist,
                train_subset_uttlist, config, self.num_jobs)

        # Initialize neural net
        print('Beginning DNN training...')
        stddev = float(1.0/config.pnorm_input_dim**0.5)
        online_preconditioning_opts = 'alpha={} num-samples-history={} update-period={} rank-in={} rank-out={} max-change-per-sample={}'.format(config.alpha, config.num_samples_history, config.update_period, config.precondition_rank_in, config.precondition_rank_out, config.max_change_per_sample)
        nnet_config_path = os.path.join(directory, 'nnet.config')
        hidden_config_path = os.path.join(directory, 'hidden.config')
        ivector_dim_path = os.path.join(directory, 'ivector_dim')
        with open(ivector_dim_path, 'r') as inf:
            ivector_dim = inf.read().strip()
        feat_dim = 13 + int(ivector_dim)

        with open(nnet_config_path, 'w', newline='') as nc:
            nc.write('SpliceComponent input-dim={} left-context={} right-context={} const-component-dim={}\n'.format(feat_dim, config.splice_width, config.splice_width, ivector_dim))
            nc.write('FixedAffineComponent matrix={}\n'.format(lda_mat_path))
            nc.write('AffineComponentPreconditionedOnline input-dim={} output-dim={} {} learning-rate={} param-stddev={} bias-stddev={}\n'.format(lda_dim, config.pnorm_input_dim, online_preconditioning_opts, config.initial_learning_rate, stddev, config.bias_stddev))
            nc.write('PnormComponent input-dim={} output-dim={} p={}\n'.format(config.pnorm_input_dim, config.pnorm_output_dim, config.p))
            nc.write('NormalizeComponent dim={}\n'.format(config.pnorm_output_dim))
            nc.write('AffineComponentPreconditionedOnline input-dim={} output-dim={} {} learning-rate={} param-stddev={} bias-stddev={}\n'.format(config.pnorm_output_dim, num_leaves, online_preconditioning_opts, config.initial_learning_rate, stddev, config.bias_stddev))
            nc.write('SoftmaxComponent dim={}\n'.format(num_leaves))

        with open(hidden_config_path, 'w', newline='') as nc:
            nc.write('AffineComponentPreconditionedOnline input-dim={} output-dim={} {} learning-rate={} param-stddev={} bias-stddev={}\n'.format(config.pnorm_output_dim, config.pnorm_input_dim, online_preconditioning_opts, config.initial_learning_rate, stddev, config.bias_stddev))
            nc.write('PnormComponent input-dim={} output-dim={} p={}\n'.format(config.pnorm_input_dim, config.pnorm_output_dim, config.p))
            nc.write('NormalizeComponent dim={}\n'.format(config.pnorm_output_dim))

        log_path = os.path.join(directory, 'log', 'nnet_init.log')
        nnet_info_path = os.path.join(directory, 'log', 'nnet_info.log')
        with open(log_path, 'w') as logf:
            with open(nnet_info_path, 'w') as outf:
                nnet_am_init_proc = subprocess.Popen([thirdparty_binary('nnet-am-init'),
                                                     os.path.join(align_directory, 'tree'),
                                                     topo_path,
                                                     "{} {} -|".format(thirdparty_binary('nnet-init'),
                                                                       nnet_config_path),
                                                    os.path.join(directory, '0.mdl')],
                                                    stderr=logf)
                nnet_am_init_proc.communicate()

                nnet_am_info = subprocess.Popen([thirdparty_binary('nnet-am-info'),
                                                os.path.join(directory, '0.mdl')],
                                                stdout=outf,
                                                stderr=logf)
                nnet_am_info.communicate()


        # Train transition probabilities and set priors
        #   First combine all previous alignments
        ali_files = glob.glob(os.path.join(align_directory, 'ali.*'))
        prev_ali_path = os.path.join(directory, 'prev_ali.')
        with open(prev_ali_path, 'wb') as outfile:
            for ali_file in ali_files:
                with open(os.path.join(align_directory, ali_file), 'rb') as infile:
                    for line in infile:
                        outfile.write(line)
        nnet_train_trans(directory, align_directory, prev_ali_path, self.num_jobs)

        # Get iteration at which we will mix up
        num_tot_iters = config.num_epochs * config.iters_per_epoch
        finish_add_layers_iter = config.num_hidden_layers * config.add_layers_period
        first_modify_iter = finish_add_layers_iter + config.add_layers_period
        mix_up_iter = (num_tot_iters + finish_add_layers_iter)/2

        # Get iterations at which we will realign
        realign_iters = []
        if config.realign_times != 0:
            div = config.realign_times + 1 # (e.g. realign 2 times = iterations split into 3 sets)
            step = num_tot_iters / div
            for i in range(0, num_tot_iters, step):
                if i == 0 or i == num_tot_iters - 1:
                    realign_iters.append(i)

        # Training loop
        for i in range(num_tot_iters):
            model_path = os.path.join(directory, '{}.mdl'.format(i))
            next_model_path = os.path.join(directory, '{}.mdl'.format(i + 1))

            # Combine all examples (could integrate validation diagnostics, etc., later-- see egs functions)
            egs_files = []
            for file in os.listdir(egs_directory):
                if file.startswith('egs'):
                    egs_files.append(file)
            with open(os.path.join(egs_directory, 'all_egs.egs'), 'wb') as outfile:
                for egs_file in egs_files:
                    with open(os.path.join(egs_directory, egs_file), 'rb') as infile:
                        for line in infile:
                            outfile.write(line)

            # Get accuracy rates for the current iteration (to pull out graph later)
            #compute_prob(i, directory, egs_directory, model_path, self.num_jobs)
            log_path = os.path.join(directory, 'log', 'compute_prob_train.{}.log'.format(i))
            with open(log_path, 'w') as logf:
                compute_prob_proc = subprocess.Popen([thirdparty_binary('nnet-compute-prob'),
                                                     model_path,
                                                     'ark:{}/all_egs.egs'.format(egs_directory)],
                                                     stdout=subprocess.PIPE,
                                                     stderr=logf)
                log_prob = compute_prob_proc.stdout.read().decode('utf-8').strip()
                compute_prob_proc.communicate()

            print("Iteration {} of {} \t\t Log-probability: {}".format(i+1, num_tot_iters, log_prob))

            # Pull out and save graphs
            # This is not quite working when done automatically - to be worked out with unit testing.
            #get_accuracy_graph(os.path.join(directory, 'log'), os.path.join(directory, 'log'))

            # If it is NOT the first iteration,
            # AND we still have layers to add,
            # AND it's the right time to add a layer...
            if i > 0 and i <= ((config.num_hidden_layers-1)*config.add_layers_period) and ((i-1)%config.add_layers_period) == 0:
                # Add a new hidden layer
                mdl = os.path.join(directory, 'tmp{}.mdl'.format(i))
                log_path = os.path.join(directory, 'log', 'temp_mdl.{}.log'.format(i))
                with open(log_path, 'w') as logf:
                    with open(mdl, 'w') as outf:
                        tmp_mdl_init_proc = subprocess.Popen([thirdparty_binary('nnet-init'),
                                                            '--srand={}'.format(i),
                                                            os.path.join(directory, 'hidden.config'),
                                                            '-'],
                                                            stdout=subprocess.PIPE,
                                                            stderr=logf)
                        tmp_mdl_ins_proc = subprocess.Popen([thirdparty_binary('nnet-insert'),
                                                            os.path.join(directory, '{}.mdl'.format(i)),
                                                            '-', '-'],
                                                            stdin=tmp_mdl_init_proc.stdout,
                                                            stdout=outf,
                                                            stderr=logf)
                        tmp_mdl_ins_proc.communicate()

            # Otherwise just use the past model
            else:
                mdl = os.path.join(directory, '{}.mdl'.format(i))

            # Shuffle examples and train nets with SGD
            nnet_train(directory, egs_directory, mdl, i, self.num_jobs)

            # Get nnet list from the various jobs on this iteration
            nnets_list = [os.path.join(directory, '{}.{}.mdl'.format((i+1), x))
                         for x in range(self.num_jobs)]

            if (i+1) >= num_tot_iters:
                learning_rate = config.final_learning_rate
            else:
                learning_rate = config.initial_learning_rate * math.exp(i * math.log(config.final_learning_rate/config.initial_learning_rate)/num_tot_iters)

            log_path = os.path.join(directory, 'log', 'average.{}.log'.format(i))
            with open(log_path, 'w') as logf:
                nnet_avg_proc = subprocess.Popen([thirdparty_binary('nnet-am-average')]
                                                 + nnets_list
                                                 + ['-'],
                                                 stdout=subprocess.PIPE,
                                                 stderr=logf)
                nnet_copy_proc = subprocess.Popen([thirdparty_binary('nnet-am-copy'),
                                                  '--learning-rate={}'.format(learning_rate),
                                                  '-',
                                                  os.path.join(directory, '{}.mdl'.format(i+1))],
                                                  stdin=nnet_avg_proc.stdout,
                                                  stderr=logf)
                nnet_copy_proc.communicate()

            # If it's the right time, do mixing up
            if config.mix_up > 0 and i == mix_up_iter:
                log_path = os.path.join(directory, 'log', 'mix_up.{}.log'.format(i))
                with open(log_path, 'w') as logf:
                    nnet_am_mixup_proc = subprocess.Popen([thirdparty_binary('nnet-am-mixup'),
                                                          '--min-count=10',
                                                          '--num-mixtures={}'.format(config.mix_up),
                                                          os.path.join(directory, '{}.mdl'.format(i+1)),
                                                          os.path.join(directory, '{}.mdl'.format(i+1))],
                                                          stderr=logf)
                    nnet_am_mixup_proc.communicate()

            # Realign if it's the right time
            if i in realign_iters:
                prev_egs_directory = egs_directory
                egs_directory = os.path.join(directory, 'egs_{}'.format(i))
                os.makedirs(egs_directory, exist_ok=True)

                #   Get average posterior for purposes of adjusting priors
                get_average_posteriors(i, directory, prev_egs_directory, config, self.num_jobs)
                log_path = os.path.join(directory, 'log', 'vector_sum_exterior.{}.log'.format(i))
                vectors_to_sum = glob.glob(os.path.join(directory, 'post.{}.*.vec'.format(i)))

                with open(log_path, 'w') as logf:
                    vector_sum_proc = subprocess.Popen([thirdparty_binary('vector-sum')]
                                                       + vectors_to_sum
                                                       + [os.path.join(directory, 'post.{}.vec'.format(i))
                                                       ],
                                                       stderr=logf)
                    vector_sum_proc.communicate()

                #   Readjust priors based on computed posteriors
                log_path = os.path.join(directory, 'log', 'adjust_priors.{}.log'.format(i))
                with open(log_path, 'w') as logf:
                    nnet_adjust_priors_proc = subprocess.Popen([thirdparty_binary('nnet-adjust-priors'),
                                                               os.path.join(directory, '{}.mdl'.format(i)),
                                                               os.path.join(directory, 'post.{}.vec'.format(i)),
                                                               os.path.join(directory, '{}.mdl'.format(i))],
                                                               stderr=logf)
                    nnet_adjust_priors_proc.communicate()

                #   Realign:
                #       Compile train graphs (gets fsts.{} for alignment)
                compile_train_graphs(directory, self.dictionary.output_directory,
                                     self.corpus.split_directory, self.num_jobs)

                #       Get alignment feats
                nnet_get_align_feats(directory, self.corpus.split_directory, fixed_ivector_dir, config, self.num_jobs)

                #       Do alignment
                nnet_align(i, directory,
                      self.dictionary.optional_silence_csl,
                      self.num_jobs, config)

                #     Finally, relabel the egs
                ali_files = glob.glob(os.path.join(directory, 'ali.*'))
                alignments = os.path.join(directory, 'alignments.')
                with open(alignments, 'wb') as outfile:
                    for ali_file in ali_files:
                        with open(os.path.join(directory, ali_file), 'rb') as infile:
                            for line in infile:
                                outfile.write(line)
                relabel_egs(i, directory, prev_egs_directory, alignments, egs_directory, self.num_jobs)


        # Rename the final model
        shutil.copy(os.path.join(directory, '{}.mdl'.format(num_tot_iters-1)), os.path.join(directory, 'final.mdl'))

        # Compile train graphs (gets fsts.{} for alignment)
        compile_train_graphs(directory, self.dictionary.output_directory,
                             self.corpus.split_directory, self.num_jobs)

        # Get alignment feats
        nnet_get_align_feats(directory, self.corpus.split_directory, fixed_ivector_dir, config, self.num_jobs)

        # Do alignment
        nnet_align("final", directory,
              self.dictionary.optional_silence_csl,
              self.num_jobs, config, mdl=os.path.join(directory, 'final.mdl'))

    def _extract_ivectors(self):
        '''
        Extracts i-vectors from a corpus using the trained i-vector extractor.
        '''
        print('Extracting i-vectors...')

        log_dir = os.path.join(self.extracted_ivector_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)

        # To do still for release: maybe add arguments to command line to tell MFA which
        # i-vector extractor to use.

        directory = self.extracted_ivector_directory

        # Only one option for now - make this an argument eventually.
        # Librispeech 100 chosen because of large number of speakers, not necessarily longer length. 
        # Thesis results tentatively confirmed that more speakers in ivector extractor => better results.
        ivector_extractor = IvectorExtractor(os.path.join(os.path.dirname(__file__), '../../pretrained_models/ls_100_ivector_extractor.zip'))
        ivector_extractor_directory = os.path.join(self.temp_directory, 'ivector_extractor')
        ivector_extractor.export_ivector_extractor(ivector_extractor_directory)

        split_dir = self.corpus.split_directory
        train_dir = self.corpus.output_directory
        config = self.ivector_extractor_config
        training_directory = self.corpus.output_directory

        # To make a directory for corpus with just 2 utterances per speaker
        # (left commented out in case we ever decide to do this)
        """max2_dir = os.path.join(directory, 'max2')
        os.makedirs(max2_dir, exist_ok=True)
        mfa_working_dir = os.getcwd()
        os.chdir("/Users/mlml/Documents/Project/kaldi2/egs/wsj/s5/steps/online/nnet2")
        copy_data_sh = "/Users/mlml/Documents/Project/kaldi2/egs/wsj/s5/steps/online/nnet2/copy_data_dir.sh"
        log_path = os.path.join(directory, 'log', 'max2.log')
        with open(log_path, 'w') as logf:
            command = [copy_data_sh, '--utts-per-spk-max', '2', train_dir, max2_dir]
            max2_proc = subprocess.Popen(command,
                                         stderr=logf)
            max2_proc.communicate()
        os.chdir(mfa_working_dir)"""

        # Write a "cmvn config" file (this is blank in the actual kaldi code, but it needs the argument passed)
        cmvn_config = os.path.join(directory, 'online_cmvn.conf')
        with open(cmvn_config, 'w') as cconf:
            cconf.write("")

        # Write a "splice config" file
        splice_config = os.path.join(directory, 'splice.conf')
        with open(splice_config, 'w') as sconf:
            sconf.write(config.splice_opts[0])
            sconf.write('\n')
            sconf.write(config.splice_opts[1])

        # Write a "config" file to input to the extraction binary
        ext_config = os.path.join(directory, 'ivector_extractor.conf')
        with open(ext_config, 'w') as ieconf:
            ieconf.write('--cmvn-config={}\n'.format(cmvn_config))
            ieconf.write('--ivector-period={}\n'.format(config.ivector_period))
            ieconf.write('--splice-config={}\n'.format(splice_config))
            ieconf.write('--lda-matrix={}\n'.format(os.path.join(ivector_extractor_directory, 'final.mat')))
            ieconf.write('--global-cmvn-stats={}\n'.format(os.path.join(ivector_extractor_directory, 'global_cmvn.stats')))
            ieconf.write('--diag-ubm={}\n'.format(os.path.join(ivector_extractor_directory, 'final.dubm')))
            ieconf.write('--ivector-extractor={}\n'.format(os.path.join(ivector_extractor_directory, 'final.ie')))
            ieconf.write('--num-gselect={}\n'.format(config.num_gselect))
            ieconf.write('--min-post={}\n'.format(config.min_post))
            ieconf.write('--posterior-scale={}\n'.format(config.posterior_scale))
            ieconf.write('--max-remembered-frames=1000\n')
            ieconf.write('--max-count={}\n'.format(0))

        # Extract i-vectors
        extract_ivectors(directory, training_directory, ext_config, config, self.num_jobs)

        # Combine i-vectors across jobs
        file_list = []
        for j in range(self.num_jobs):
            file_list.append(os.path.join(directory, 'ivector_online.{}.scp'.format(j)))

        with open(os.path.join(directory, 'ivector_online.scp'), 'w') as outfile:
            for fname in file_list:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)
