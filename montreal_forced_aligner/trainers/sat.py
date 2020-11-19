import os
from tqdm import tqdm
import subprocess
import shutil

from ..multiprocessing import (align, compile_train_graphs,
                               acc_stats, tree_stats, convert_alignments,
                               calc_fmllr, compute_alignment_improvement)
from ..helper import thirdparty_binary, make_path_safe

from .triphone import TriphoneTrainer


class SatTrainer(TriphoneTrainer):
    '''

    Configuration class for speaker adapted training (SAT)

    Attributes
    ----------
    num_iterations : int
        Number of training iterations to perform, defaults to 40
    transition_scale : float
        Scaling of transition costs in alignment, defaults to 1.0
    acoustic_scale : float
        Scaling of acoustic costs in alignment, defaults to 0.1
    self_loop_scale : float
        Scaling of self loop costs in alignment, defaults to 0.1
    beam : int
        Default beam width for alignment, defaults = 10
    retry_beam : int
        Beam width to fall back on if no alignment is produced, defaults to 40
    max_gaussians : int
        Total number of gaussians, defaults to 1000
    boost_silence : float
        Factor by which to boost silence likelihoods in alignment, defaults to 1.0
    realignment_iterations : list
        List of iterations to perform alignment
    power : float
        Exponent for number of gaussians according to occurrence counts, defaults to 0.25
    num_leaves : int
        Number of states in the decision tree, defaults to 1000
    max_gaussians : int
        Number of gaussians in the decision tree, defaults to 10000
    cluster_threshold : int
        For build-tree control final bottom-up clustering of leaves, defaults to 100
    fmllr_update_type : str
        Type of fMLLR estimation, defaults to ``'full'``
    fmllr_iterations : list
        List of iterations to perform fMLLR estimation
    silence_weight : float
        Weight on silence in fMLLR estimation
    '''

    def __init__(self, default_feature_config):
        super(SatTrainer, self).__init__(default_feature_config)
        self.fmllr_update_type = 'full'
        self.fmllr_iterations = []
        max_fmllr_iter = int(self.num_iterations/2) - 1
        for i in range(1, max_fmllr_iter):
            if i < max_fmllr_iter /2 and i % 2 == 0:
                self.fmllr_iterations.append(i)
        self.fmllr_iterations.append(max_fmllr_iter)
        self.silence_weight = 0.0
        self.feature_config.fmllr = True

    def compute_calculated_properties(self):
        super(SatTrainer, self).compute_calculated_properties()
        self.fmllr_iterations = []
        max_fmllr_iter = int(self.num_iterations/2) - 1
        for i in range(1, max_fmllr_iter):
            if i < max_fmllr_iter /2 and i % 2 == 0:
                self.fmllr_iterations.append(i)
        self.fmllr_iterations.append(max_fmllr_iter)

    @property
    def train_type(self):
        return 'sat'

    def train(self, call_back=None):
        final_mdl_path = os.path.join(self.train_directory, 'final.mdl')
        if os.path.exists(final_mdl_path):
            print('{} training already done, skipping.'.format(self.identifier))
            return
        num_gauss = self.initial_gaussians
        if call_back == print:
            iters = tqdm(range(1, self.num_iterations))
        else:
            iters = range(1, self.num_iterations)
        sil_phones = self.dictionary.silence_csl
        for i in iters:
            model_path = os.path.join(self.train_directory, '{}.mdl'.format(i))
            occs_path = os.path.join(self.train_directory, '{}.occs'.format(i + 1))
            next_model_path = os.path.join(self.train_directory, '{}.mdl'.format(i + 1))
            if os.path.exists(next_model_path):
                continue
            if i in self.realignment_iterations:
                align(i, self.train_directory, self.data_directory,
                      self.dictionary.optional_silence_csl,
                      self.corpus.num_jobs, self)
                if self.debug:
                    compute_alignment_improvement(i, self, self.train_directory, self.corpus.num_jobs)
            if i in self.fmllr_iterations:
                calc_fmllr(self.train_directory, self.data_directory, sil_phones,
                           self.corpus.num_jobs, self, initial=False, iteration=i)

            acc_stats(i, self.train_directory, self.data_directory, self.corpus.num_jobs, self)
            log_path = os.path.join(self.log_directory, 'update.{}.log'.format(i))
            with open(log_path, 'w') as logf:
                acc_files = [os.path.join(self.train_directory, '{}.{}.acc'.format(i, x))
                             for x in range(self.corpus.num_jobs)]
                est_proc = subprocess.Popen([thirdparty_binary('gmm-est'),
                                             '--write-occs=' + occs_path,
                                             '--mix-up=' + str(num_gauss), '--power=' + str(self.power),
                                             model_path,
                                             "{} - {}|".format(thirdparty_binary('gmm-sum-accs'),
                                                               ' '.join(map(make_path_safe, acc_files))),
                                             next_model_path],
                                            stderr=logf)
                est_proc.communicate()
            if not os.path.exists(next_model_path):
                raise(Exception('There was an error training in iteration {}, please check the logs.'.format(i)))
            if not self.debug:
                for f in acc_files:
                    os.remove(f)
            self.parse_log_directory(self.log_directory, i, self.corpus.num_jobs, call_back)
            if i < self.final_gaussian_iteration:
                num_gauss += self.gaussian_increment
        shutil.copy(os.path.join(self.train_directory, '{}.mdl'.format(self.num_iterations)),
                    os.path.join(self.train_directory, 'final.mdl'))
        shutil.copy(os.path.join(self.train_directory, '{}.occs'.format(self.num_iterations)),
                    os.path.join(self.train_directory, 'final.occs'))
        if not self.debug:
            for i in range(1, self.num_iterations):
                model_path = os.path.join(self.train_directory, '{}.mdl'.format(i))
                try:
                    os.remove(model_path)
                except FileNotFoundError:
                    pass
                try:
                    os.remove(os.path.join(self.train_directory, '{}.occs'.format(i)))
                except FileNotFoundError:
                    pass

    def init_training(self, identifier, temporary_directory, corpus, dictionary, previous_trainer):
        self._setup_for_init(identifier, temporary_directory, corpus, dictionary)

        if os.path.exists(os.path.join(self.train_directory, '1.mdl')):
            return
        self.feature_config.fmllr = True

        print('Initializing speaker-adapted triphone training...')
        align_directory = previous_trainer.align_directory
        context_opts = []
        ci_phones = self.dictionary.silence_csl

        tree_stats(self.train_directory, align_directory,
                   self.data_directory, ci_phones, self.corpus.num_jobs, self)
        log_path = os.path.join(self.log_directory, 'questions.log')
        tree_path = os.path.join(self.train_directory, 'tree')
        treeacc_path = os.path.join(self.train_directory, 'treeacc')
        sets_int_path = os.path.join(self.dictionary.phones_dir, 'sets.int')
        roots_int_path = os.path.join(self.dictionary.phones_dir, 'roots.int')
        extra_question_int_path = os.path.join(self.dictionary.phones_dir, 'extra_questions.int')
        topo_path = os.path.join(self.dictionary.output_directory, 'topo')
        questions_path = os.path.join(self.train_directory, 'questions.int')
        questions_qst_path = os.path.join(self.train_directory, 'questions.qst')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('cluster-phones')] + context_opts +
                            [treeacc_path, sets_int_path, questions_path], stderr=logf)

        with open(extra_question_int_path, 'r') as inf, \
                open(questions_path, 'a') as outf:
            for line in inf:
                outf.write(line)

        log_path = os.path.join(self.log_directory, 'compile_questions.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('compile-questions')] + context_opts +
                            [topo_path, questions_path, questions_qst_path],
                            stderr=logf)

        log_path = os.path.join(self.log_directory, 'build_tree.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('build-tree')] + context_opts +
                            ['--verbose=1', '--max-leaves={}'.format(self.initial_gaussians),
                             '--cluster-thresh={}'.format(self.cluster_threshold),
                             treeacc_path, roots_int_path, questions_qst_path,
                             topo_path, tree_path], stderr=logf)

        log_path = os.path.join(self.log_directory, 'init_model.log')
        occs_path = os.path.join(self.train_directory, '0.occs')
        mdl_path = os.path.join(self.train_directory, '0.mdl')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('gmm-init-model'),
                             '--write-occs=' + occs_path, tree_path, treeacc_path,
                             topo_path, mdl_path], stderr=logf)

        log_path = os.path.join(self.log_directory, 'mixup.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('gmm-mixup'),
                             '--mix-up={}'.format(self.initial_gaussians),
                             mdl_path, occs_path, mdl_path], stderr=logf)
        os.remove(treeacc_path)

        compile_train_graphs(self.train_directory, self.dictionary.output_directory,
                             self.data_directory, self.corpus.num_jobs)
        os.rename(occs_path, os.path.join(self.train_directory, '1.occs'))
        os.rename(mdl_path, os.path.join(self.train_directory, '1.mdl'))

        convert_alignments(self.train_directory, align_directory, self.corpus.num_jobs)

        calc_fmllr(self.train_directory, self.data_directory,
                   self.dictionary.silence_csl, self.corpus.num_jobs, self, initial=True)

        if os.path.exists(os.path.join(align_directory, 'trans.0')):
            for i in range(self.corpus.num_jobs):
                shutil.copy(os.path.join(align_directory, 'trans.{}'.format(i)),
                            os.path.join(self.train_directory, 'trans.{}'.format(i)))
        print('Initialization complete!')

