import os
from tqdm import tqdm
import subprocess
import shutil
import time

from ..multiprocessing import (align, acc_stats, calc_lda_mllt, lda_acc_stats, compute_alignment_improvement)
from ..helper import thirdparty_binary, make_path_safe, log_kaldi_errors
from ..exceptions import KaldiProcessingError
from .triphone import TriphoneTrainer


class LdaTrainer(TriphoneTrainer):
    """

    Configuration class for LDA+MLLT training

    Attributes
    ----------
    lda_dimension : int
        Dimensionality of the LDA matrix
    mllt_iterations : list
        List of iterations to perform MLLT estimation
    random_prune : float
        This is approximately the ratio by which we will speed up the
        LDA and MLLT calculations via randomized pruning
    """

    def __init__(self, default_feature_config):
        super(LdaTrainer, self).__init__(default_feature_config)
        self.lda_dimension = 40
        self.mllt_iterations = []
        max_mllt_iter = int(self.num_iterations / 2) - 1
        for i in range(1, max_mllt_iter):
            if i < max_mllt_iter / 2 and i % 2 == 0:
                self.mllt_iterations.append(i)
        self.mllt_iterations.append(max_mllt_iter)
        if not self.mllt_iterations:
            self.mllt_iterations = range(1,4)
        self.random_prune = 4.0

        self.feature_config.lda = True
        self.feature_config.deltas = True

    def compute_calculated_properties(self):
        super(LdaTrainer, self).compute_calculated_properties()
        self.mllt_iterations = []
        max_mllt_iter = int(self.num_iterations / 2) - 1
        for i in range(1, max_mllt_iter):
            if i < max_mllt_iter / 2 and i % 2 == 0:
                self.mllt_iterations.append(i)
        self.mllt_iterations.append(max_mllt_iter)

    @property
    def train_type(self):
        return 'lda'

    @property
    def lda_options(self):
        return {'lda_dimension': self.lda_dimension, 'boost_silence': self.boost_silence,
                'random_prune': self.random_prune}

    def init_training(self, identifier, temporary_directory, corpus, dictionary, previous_trainer):
        self._setup_for_init(identifier, temporary_directory, corpus, dictionary)
        done_path = os.path.join(self.train_directory, 'done')
        dirty_path = os.path.join(self.train_directory, 'dirty')
        if os.path.exists(done_path):
            self.logger.info('{} training already done, skipping initialization.'.format(self.identifier))
            return
        begin = time.time()
        try:
            self.feature_config.directory = None
            lda_acc_stats(self.train_directory, self.data_directory, previous_trainer.align_directory, self,
                          self.dictionary.silence_csl, self.corpus.num_jobs)
            self.feature_config.directory = self.train_directory
            if self.data_directory != self.corpus.split_directory():
                utt_list = []
                subset_utt_path = os.path.join(self.data_directory, 'included_utts.txt')
                with open(subset_utt_path, 'r') as f:
                    for line in f:
                        utt_list.append(line.strip())
        except Exception as e:
            with open(dirty_path, 'w') as _:
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise
        self._setup_tree(previous_trainer.align_directory)
        self.logger.info('Initialization complete!')
        self.logger.debug('Initialization took {} seconds'.format(time.time() - begin))

    def train(self, call_back=None):
        done_path = os.path.join(self.train_directory, 'done')
        dirty_path = os.path.join(self.train_directory, 'dirty')
        if os.path.exists(done_path):
            self.logger.info('{} training already done, skipping.'.format(self.identifier))
            return
        begin = time.time()
        try:
            num_gauss = self.initial_gaussians
            if call_back == print:
                iters = tqdm(range(0, self.num_iterations))
            else:
                iters = range(0, self.num_iterations)
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
                if i in self.mllt_iterations:
                    calc_lda_mllt(self.train_directory, self.data_directory,  sil_phones,
                                  self.corpus.num_jobs, self,
                                  initial=False, iteration=i)

                acc_stats(i, self.train_directory, self.data_directory, self.corpus.num_jobs,
                          self)
                log_path = os.path.join(self.log_directory, 'update.{}.log'.format(i))
                with open(log_path, 'w') as log_file:
                    acc_files = [os.path.join(self.train_directory, '{}.{}.acc'.format(i, x))
                                 for x in range(self.corpus.num_jobs)]
                    est_proc = subprocess.Popen([thirdparty_binary('gmm-est'),
                                                 '--write-occs=' + occs_path,
                                                 '--mix-up=' + str(num_gauss), '--power=' + str(self.power),
                                                 model_path,
                                                 "{} - {}|".format(thirdparty_binary('gmm-sum-accs'),
                                                                   ' '.join(map(make_path_safe, acc_files))),
                                                 next_model_path],
                                                stderr=log_file)
                    est_proc.communicate()
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
            shutil.copy(os.path.join(self.train_directory, 'lda.mat'),
                        os.path.join(self.align_directory, 'lda.mat'))
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
        except Exception as e:
            with open(dirty_path, 'w'):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise
        with open(done_path, 'w'):
            pass

        self.logger.info('Training complete!')
        self.logger.debug('Training took {} seconds'.format(time.time() - begin))
