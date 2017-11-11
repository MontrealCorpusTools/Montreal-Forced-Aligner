import os
import shutil
import subprocess
import re
from tqdm import tqdm

from ..helper import thirdparty_binary, make_path_safe

from ..config import MonophoneConfig, TriphoneConfig, TriphoneFmllrConfig

from ..multiprocessing import (align, mono_align_equal, compile_train_graphs,
                               acc_stats, tree_stats, convert_alignments,
                               convert_ali_to_textgrids, calc_fmllr)

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
                 tri_fmllr_params=None, debug=False,
                 skip_input=False):
        if mono_params is None:
            mono_params = {}
        if tri_params is None:
            tri_params = {}
        if tri_fmllr_params is None:
            tri_fmllr_params = {}
        self.mono_config = MonophoneConfig(**mono_params)
        self.tri_config = TriphoneConfig(**tri_params)
        self.tri_fmllr_config = TriphoneFmllrConfig(**tri_fmllr_params)
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

    def _align_si(self, fmllr=True):
        '''
        Generate an alignment of the dataset
        '''
        if fmllr and os.path.exists(self.tri_fmllr_final_model_path):
            model_directory = self.tri_fmllr_directory
            output_directory = self.tri_fmllr_ali_directory
            config = self.tri_fmllr_config
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
        os.makedirs(log_dir, exist_ok=True)

        shutil.copy(os.path.join(model_directory, 'tree'), output_directory)
        shutil.copyfile(os.path.join(model_directory, 'final.mdl'),
                        os.path.join(output_directory, '0.mdl'))

        shutil.copyfile(os.path.join(model_directory, 'final.occs'),
                        os.path.join(output_directory, '0.occs'))

        feat_type = 'delta'

        compile_train_graphs(output_directory, self.dictionary.output_directory,
                             self.corpus.split_directory, self.num_jobs, debug=self.debug)
        align(0, output_directory, self.corpus.split_directory,
              optional_silence, self.num_jobs, config)
        shutil.copyfile(os.path.join(output_directory, '0.mdl'), os.path.join(output_directory, 'final.mdl'))
        shutil.copyfile(os.path.join(output_directory, '0.occs'), os.path.join(output_directory, 'final.occs'))

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
        model_directory = self.tri_directory
        output_directory = self.tri_ali_directory
        self._align_si(fmllr=False)
        sil_phones = self.dictionary.silence_csl

        log_dir = os.path.join(output_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)

        calc_fmllr(output_directory, self.corpus.split_directory,
                   sil_phones, self.num_jobs, self.tri_fmllr_config, initial=True)
        optional_silence = self.dictionary.optional_silence_csl
        align(0, output_directory, self.corpus.split_directory,
              optional_silence, self.num_jobs, self.tri_fmllr_config)

    def _init_tri(self, fmllr=False):
        if fmllr:
            config = self.tri_fmllr_config
            directory = self.tri_fmllr_directory
            align_directory = self.tri_ali_directory
        else:
            config = self.tri_config
            directory = self.tri_directory
            align_directory = self.mono_ali_directory
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
        os.remove(treeacc_path)

        compile_train_graphs(directory, self.dictionary.output_directory,
                             self.corpus.split_directory, self.num_jobs)
        os.rename(occs_path, os.path.join(directory, '1.occs'))
        os.rename(mdl_path, os.path.join(directory, '1.mdl'))

        convert_alignments(directory, align_directory, self.num_jobs)

        if os.path.exists(os.path.join(align_directory, 'trans.0')):
            for i in range(self.num_jobs):
                shutil.copy(os.path.join(align_directory, 'trans.{}'.format(i)),
                            os.path.join(directory, 'trans.{}'.format(i)))

    def train_tri_fmllr(self):
        '''
        Perform speaker-adapted triphone training
        '''
        if os.path.exists(self.tri_fmllr_final_model_path):
            print('Triphone FMLLR training already done, using previous final.mdl')
            return
        if not os.path.exists(self.tri_ali_directory):
            self._align_fmllr()

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
            if os.path.exists(next_model_path):
                continue
            if i in config.realign_iters:
                align(i, directory, self.corpus.split_directory,
                      self.dictionary.optional_silence_csl,
                      self.num_jobs, config)
            if config.do_fmllr and i in config.fmllr_iters:
                calc_fmllr(directory, self.corpus.split_directory, sil_phones,
                           self.num_jobs, config, initial=False, iteration=i)

            acc_stats(i, directory, self.corpus.split_directory, self.num_jobs,
                      config.do_fmllr)
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
