import os
import subprocess
import time
from .base import BaseTrainer
from ..helper import thirdparty_binary, log_kaldi_errors, parse_logs
from ..exceptions import KaldiProcessingError

from ..multiprocessing import compile_train_graphs, tree_stats, convert_alignments


class TriphoneTrainer(BaseTrainer):
    """
    Configuration class for triphone training

    Attributes
    ----------
    num_iterations : int
        Number of training iterations to perform, defaults to 40
    max_gaussians : int
        Total number of gaussians, defaults to 1000
    num_leaves : int
        Number of states in the decision tree, defaults to 1000
    max_gaussians : int
        Number of gaussians in the decision tree, defaults to 10000
    cluster_threshold : int
        For build-tree control final bottom-up clustering of leaves, defaults to 100
    """

    def __init__(self, default_feature_config):
        super(TriphoneTrainer, self).__init__(default_feature_config)

        self.num_iterations = 35
        self.num_leaves = 1000
        self.max_gaussians = 10000
        self.cluster_threshold = -1
        self.compute_calculated_properties()

    def compute_calculated_properties(self):
        for i in range(0, self.num_iterations, 10):
            if i == 0:
                continue
            self.realignment_iterations.append(i)
        self.initial_gaussians = self.num_leaves

    @property
    def train_type(self):
        return 'tri'

    @property
    def phone_type(self):
        return 'triphone'

    def _setup_tree(self, align_directory):
        dirty_path = os.path.join(self.train_directory, 'dirty')
        try:
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
            with open(log_path, 'w') as log_file:
                subprocess.call([thirdparty_binary('cluster-phones')] + context_opts +
                                [treeacc_path, sets_int_path, questions_path], stderr=log_file)

            with open(extra_question_int_path, 'r') as inf, \
                    open(questions_path, 'a') as outf:
                for line in inf:
                    outf.write(line)

            log_path = os.path.join(self.log_directory, 'compile_questions.log')
            with open(log_path, 'w') as log_file:
                subprocess.call([thirdparty_binary('compile-questions')] + context_opts +
                                [topo_path, questions_path, questions_qst_path],
                                stderr=log_file)

            log_path = os.path.join(self.log_directory, 'build_tree.log')
            with open(log_path, 'w') as log_file:
                subprocess.call([thirdparty_binary('build-tree')] + context_opts +
                                ['--verbose=1', '--max-leaves={}'.format(self.initial_gaussians),
                                 '--cluster-thresh={}'.format(self.cluster_threshold),
                                 treeacc_path, roots_int_path, questions_qst_path,
                                 topo_path, tree_path], stderr=log_file)

            log_path = os.path.join(self.log_directory, 'init_model.log')
            occs_path = os.path.join(self.train_directory, '0.occs')
            mdl_path = os.path.join(self.train_directory, '0.mdl')
            with open(log_path, 'w') as log_file:
                subprocess.call([thirdparty_binary('gmm-init-model'),
                                 '--write-occs=' + occs_path, tree_path, treeacc_path,
                                 topo_path, mdl_path], stderr=log_file)

            log_path = os.path.join(self.log_directory, 'mixup.log')
            with open(log_path, 'w') as log_file:
                subprocess.call([thirdparty_binary('gmm-mixup'),
                                 '--mix-up={}'.format(self.initial_gaussians),
                                 mdl_path, occs_path, mdl_path], stderr=log_file)
            #os.remove(treeacc_path)
            parse_logs(self.log_directory)

            compile_train_graphs(self.train_directory, self.dictionary.output_directory,
                                 self.data_directory, self.corpus.num_jobs, self)
            os.rename(occs_path, os.path.join(self.train_directory, '1.occs'))
            os.rename(mdl_path, os.path.join(self.train_directory, '1.mdl'))

            convert_alignments(self.train_directory, align_directory, self.corpus.num_jobs, self)
        except Exception as e:
            with open(dirty_path, 'w'):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise

    def init_training(self, identifier, temporary_directory, corpus, dictionary, previous_trainer):
        self._setup_for_init(identifier, temporary_directory, corpus, dictionary)
        done_path = os.path.join(self.train_directory, 'done')
        if os.path.exists(done_path):
            self.logger.info('{} training already done, skipping initialization.'.format(self.identifier))
            return
        begin = time.time()
        align_directory = previous_trainer.align_directory
        self._setup_tree(align_directory)

        self.logger.info('Initialization complete!')
        self.logger.debug('Initialization took {} seconds'.format(time.time() - begin))
