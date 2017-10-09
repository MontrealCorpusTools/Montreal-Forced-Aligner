import os
import shutil
import subprocess
import re
from tqdm import tqdm

from ..helper import thirdparty_binary, make_path_safe

from ..config import (MonophoneConfig, TriphoneConfig, TriphoneFmllrConfig,
                      LdaMlltConfig, DiagUbmConfig, iVectorExtractorConfig,
                      NnetBasicConfig)

from ..multiprocessing import (align, mono_align_equal, compile_train_graphs,
                               acc_stats, tree_stats, convert_alignments,
                               convert_ali_to_textgrids, calc_fmllr,
                               calc_lda_mllt)

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
                 debug=False):
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
        if self.corpus.num_jobs != num_jobs:
            num_jobs = self.corpus.num_jobs
        self.num_jobs = num_jobs
        if not temp_directory:
            temp_directory = TEMP_DIR
        self.temp_directory = temp_directory
        self.call_back = call_back
        if self.call_back is None:
            self.call_back = print
        self.verbose = False
        self.debug = debug
        self.setup()

    def setup(self):
        self.dictionary.write()
        self.corpus.initialize_corpus(self.dictionary)
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
        return os.path.join(self.diag_ubm_directory, 'final.dubm')   # May not be this label (but I think so)

    @property
    def ivector_extractor_directory(self):
        return os.path.join(self.temp_directory, 'ivector_extractor')

    @property
    def ivector_extractor_final_model_path(self):
        return os.path.join(self.ivector_extractor_directory, 'final.ie')

    @property
    def nnet_basic_directory(self):
        return os.path.join(self.temp_directory, 'nnet_basic')

    @property
    def nnet_basic_final_model_path(self):
        return os.path.join(self.nnet_basic_directory, 'final.mdl')

    # End of nnet properties

    def export_textgrids(self):
        '''
        Export a TextGrid file for every sound file in the dataset
        '''
        if os.path.exists(self.tri_fmllr_final_model_path):
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
            print("here1 fmllr")
            model_directory = self.tri_fmllr_directory
            #model_directory = self.tri_directory
            output_directory = self.tri_fmllr_ali_directory
            config = self.tri_fmllr_config
        #
        #elif lda_mllt and os.path.exists(self.lda_mllt_final_model_path):

        elif fmllr:     # First pass with fmllr, final path doesn't exist yet
            print("here2 fmllr first pass")
            model_directory = self.tri_directory
            output_directory = self.tri_fmllr_ali_directory
            #output_directory = self.tri_fmllr_directory
            config = self.tri_fmllr_config

        elif lda_mllt and os.path.exists(self.lda_mllt_final_model_path):
            print("here lda_mllt")
            model_directory = self.lda_mllt_directory
            output_directory = self.lda_mllt_ali_directory
            config = self.lda_mllt_config
        #
        elif lda_mllt:  # First pass with LDA + MLLT, final path doesn't exist yet
            print("first pass lda_mllt")
            model_directory = self.tri_fmllr_directory
            #output_directory = self.lda_mllt_ali_directory
            output_directory = self.lda_mllt_ali_directory
            config = self.lda_mllt_config
        elif os.path.exists(self.tri_final_model_path):
            print("here2 no fmllr")
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

        print("FEATURE NAME PASSING TO ALIGN:", feature_name)
        align(0, output_directory, self.corpus.split_directory,
              optional_silence, self.num_jobs, config, feature_name=feature_name)
        shutil.copyfile(os.path.join(output_directory, '0.mdl'), os.path.join(output_directory, 'final.mdl'))
        shutil.copyfile(os.path.join(output_directory, '0.occs'), os.path.join(output_directory, 'final.occs'))

        print("checking:", os.path.exists(os.path.join(self.tri_fmllr_directory, 'final.mdl')))
        if output_directory == self.tri_fmllr_ali_directory:
            os.makedirs(self.tri_fmllr_directory) #, exists_ok=True)
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
        #model_directory = self.tri_ali_directory
        #output_directory = self.tri_ali_directory
        output_directory = self.tri_fmllr_ali_directory
        #output_directory = self.tri_fmllr_directory # End up putting fmllr alignments here
        #self._align_si(fmllr=False)
        self._align_si(fmllr=True)
        sil_phones = self.dictionary.silence_csl

        log_dir = os.path.join(output_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)

        print("calculating fmllr")
        calc_fmllr(output_directory, self.corpus.split_directory,
                   sil_phones, self.num_jobs, self.tri_fmllr_config, initial=True)
        optional_silence = self.dictionary.optional_silence_csl
        print("from align_fmllr:")
        #align(0, output_directory, self.corpus.split_directory,
        #      optional_silence, self.num_jobs, self.tri_fmllr_config)
        #print("model dir:", model_directory)
        align(0, model_directory, self.corpus.split_directory,
              optional_silence, self.num_jobs, self.tri_fmllr_config)

        #mdl_path = os.path.join(output_directory, '0.mdl')
        """mdl_path = os.path.join(model_directory, '0.mdl')
        print("0.MDL PATH CHECK:", mdl_path, os.path.exists(mdl_path))
        new_mdl_path = os.path.join(self.tri_fmllr_ali_directory, '0.mdl')
        print("NEW MDL PATH CHECK:", new_mdl_path)
        shutil.copyfile(mdl_path, new_mdl_path)"""

    def _init_tri(self, fmllr=False):
        if fmllr:
            config = self.tri_fmllr_config
            directory = self.tri_fmllr_directory
            #align_directory = self.tri_ali_directory    # The previous
            align_directory = self.tri_fmllr_ali_directory
        else:
            config = self.tri_config
            directory = self.tri_directory
            align_directory = self.mono_ali_directory
        #if os.path.exists(os.path.join(directory, '1.mdl')):
        #    return
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

        os.remove(treeacc_path)

        compile_train_graphs(directory, self.dictionary.output_directory,
                             self.corpus.split_directory, self.num_jobs)

        print("@@@@", os.path.exists(os.path.join(directory, '0.mdl')))
        print("@@@@", os.path.exists(os.path.join(directory, '0.occs')))
        #os.rename(occs_path, os.path.join(directory, '1.occs'))
        #os.rename(mdl_path, os.path.join(directory, '1.mdl'))
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
        # Commented out for testing
        #if os.path.exists(self.tri_fmllr_final_model_path):
        #    print('Triphone FMLLR training already done, using previous final.mdl')
        #    return

        #if not os.path.exists(self.tri_ali_directory):
        #    self._align_fmllr()
        print("going into align_fmllr")
        self._align_fmllr()

        os.makedirs(os.path.join(self.tri_fmllr_directory, 'log'), exist_ok=True)
        self._init_tri(fmllr=True)
        self._do_tri_fmllr_training()

    def _do_tri_fmllr_training(self):
        self.call_back('Beginning speaker-adapted triphone training...')
        self._do_training(self.tri_fmllr_directory, self.tri_fmllr_config)

    def _do_training(self, directory, config):
        print("doing training")
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
            #if os.path.exists(next_model_path):
            #    continue
            if i in config.realign_iters:
                print("from do_training")
                align(i, directory, self.corpus.split_directory,
                      self.dictionary.optional_silence_csl,
                      self.num_jobs, config,
                      feature_name='cmvnsplicetransformfeats')
                #return
            if config.do_fmllr and i in config.fmllr_iters:
                print("calc fmllr")
                calc_fmllr(directory, self.corpus.split_directory, sil_phones,
                           self.num_jobs, config, initial=False, iteration=i)
            #
            if config.do_lda_mllt and i <= config.num_iters:
                print("calc lda mllt")
                calc_lda_mllt(directory, self.corpus.split_directory,   # Could change this to make ali directory later
                #calc_lda_mllt(self.lda_mllt_ali_directory, sil_phones,
                              self.lda_mllt_directory, sil_phones,
                              self.num_jobs, config, config.num_iters,
                              initial=False, iteration=i, corpus=self.corpus)
            #
            print("getting stats")
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
        print("dir where final.mdl is:", directory)
        shutil.copy(os.path.join(directory, '{}.mdl'.format(config.num_iters)),
                    os.path.join(directory, 'final.mdl'))
        print("moving final occs:")
        shutil.copy(os.path.join(directory, '{}.occs'.format(config.num_iters)),
                    os.path.join(directory, 'final.occs'))
        #shutil.copy(os.path.join(self.lda_mllt_directory, '{}.mat'.format(config.num_iters)),
        #            os.path.join(self.lda_mllt_directory, 'final.mat'))
        print("moving final mat:")
        shutil.copy(os.path.join(directory, '{}.mat'.format(config.num_iters-1)),
                    os.path.join(directory, 'final.mat'))
