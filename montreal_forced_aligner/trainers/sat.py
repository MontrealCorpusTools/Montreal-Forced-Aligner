import os
from tqdm import tqdm
import subprocess
import shutil
import time

from ..multiprocessing import (align, compile_train_graphs,
                               acc_stats, tree_stats, convert_alignments,
                               calc_fmllr, compute_alignment_improvement, compile_information)
from ..helper import thirdparty_binary, make_path_safe, log_kaldi_errors, parse_logs, load_scp
from ..exceptions import KaldiProcessingError

from .triphone import TriphoneTrainer


class SatTrainer(TriphoneTrainer):
    """

    Configuration class for speaker adapted training (SAT)

    Attributes
    ----------
    fmllr_update_type : str
        Type of fMLLR estimation, defaults to ``'full'``
    fmllr_iterations : list
        List of iterations to perform fMLLR estimation
    silence_weight : float
        Weight on silence in fMLLR estimation
    """

    def __init__(self, default_feature_config):
        super(SatTrainer, self).__init__(default_feature_config)
        self.fmllr_update_type = 'full'
        self.fmllr_iterations = []
        max_fmllr_iter = int(self.num_iterations/2) - 1
        for i in range(1, max_fmllr_iter):
            if i < max_fmllr_iter / 2 and i % 2 == 0:
                self.fmllr_iterations.append(i)
        self.fmllr_iterations.append(max_fmllr_iter)
        self.silence_weight = 0.0
        self.feature_config.fmllr = True
        self.use_fmllr_mp = False

    def compute_calculated_properties(self):
        super(SatTrainer, self).compute_calculated_properties()
        self.fmllr_iterations = []
        max_fmllr_iter = int(self.num_iterations / 2) - 1
        for i in range(1, max_fmllr_iter):
            if i < max_fmllr_iter / 2 and i % 2 == 0:
                self.fmllr_iterations.append(i)
        self.fmllr_iterations.append(max_fmllr_iter)

    @property
    def train_type(self):
        return 'sat'

    def train(self, call_back=None):
        done_path = os.path.join(self.train_directory, 'done')
        dirty_path = os.path.join(self.train_directory, 'dirty')
        if os.path.exists(done_path):
            self.logger.info('{} training already done, skipping initialization.'.format(self.identifier))
            return
        begin = time.time()
        num_gauss = self.initial_gaussians
        if call_back == print:
            iters = tqdm(range(1, self.num_iterations))
        else:
            iters = range(1, self.num_iterations)
        sil_phones = self.dictionary.silence_csl
        try:
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
                    parse_logs(self.log_directory)
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

    def align(self, subset, call_back=None):
        dirty_path = os.path.join(self.align_directory, 'dirty')
        if os.path.exists(dirty_path):  # if there was an error, let's redo from scratch
            shutil.rmtree(self.align_directory)
        done_path = os.path.join(self.align_directory, 'done')
        if not os.path.exists(done_path):
            message = 'Generating alignments using {} models'.format(self.identifier)
            if subset:
                message += ' using {} utterances...'.format(subset)
            else:
                message += ' for the whole corpus...'
            self.logger.info(message)
            begin = time.time()
            self.logger.debug('Using {} as the feature name'.format(self.feature_file_base_name))
            if subset is None:
                align_data_directory = self.corpus.split_directory()
            else:
                align_data_directory = self.corpus.subset_directory(subset, self.feature_config)
            try:
                log_dir = os.path.join(self.align_directory, 'log')
                os.makedirs(log_dir, exist_ok=True)
                shutil.copy(os.path.join(self.train_directory, 'tree'), self.align_directory)
                shutil.copyfile(os.path.join(self.train_directory, 'final.mdl'),
                                os.path.join(self.align_directory, 'final.mdl'))

                if os.path.exists(os.path.join(self.train_directory, 'lda.mat')):
                    shutil.copyfile(os.path.join(self.train_directory, 'lda.mat'),
                                    os.path.join(self.align_directory, 'lda.mat'))
                shutil.copyfile(os.path.join(self.train_directory, 'final.occs'),
                                os.path.join(self.align_directory, 'final.occs'))
                compile_train_graphs(self.align_directory, self.dictionary.output_directory,
                                     align_data_directory, self.corpus.num_jobs, self)
                if align_data_directory == self.data_directory and os.path.exists(os.path.join(self.train_directory, 'trans.0')):
                    for i in range(self.corpus.num_jobs):
                        shutil.copy(os.path.join(self.train_directory, 'trans.{}'.format(i)),
                                    os.path.join(self.align_directory, 'trans.{}'.format(i)))
                align('final', self.align_directory, align_data_directory,
                      self.dictionary.optional_silence_csl,
                      self.corpus.num_jobs, self, self.align_directory)

                unaligned, average_log_like = compile_information(self.align_directory, self.corpus,
                                                                  self.corpus.num_jobs, self)
                self.logger.debug(f'Before SAT, average per frame likelihood (this might not actually mean anything): {average_log_like}')

                if not os.path.exists(os.path.join(self.align_directory, 'trans.0')):
                    calc_fmllr(self.align_directory, align_data_directory,
                          self.dictionary.optional_silence_csl, self.corpus.num_jobs, self, initial=True, iteration='final')
                    align('final', self.align_directory, align_data_directory,
                          self.dictionary.optional_silence_csl,
                          self.corpus.num_jobs, self, self.align_directory)
                self.save(os.path.join(self.align_directory, 'acoustic_model.zip'))

                unaligned, average_log_like = compile_information(self.align_directory, self.corpus,
                                                                  self.corpus.num_jobs, self)
                self.logger.debug(f'Following SAT, average per frame likelihood (this might not actually mean anything): {average_log_like}')
            except Exception as e:
                with open(dirty_path, 'w'):
                    pass
                if isinstance(e, KaldiProcessingError):
                    log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
                raise
            with open(done_path, 'w'):
                pass
            self.logger.debug('Alignment took {} seconds'.format(time.time() - begin))
        else:
            self.logger.info('Alignments using {} models already done'.format(self.identifier))
        if self.debug:
            self.export_textgrids()

    def init_training(self, identifier, temporary_directory, corpus, dictionary, previous_trainer):
        self.feature_config.fmllr = False
        self._setup_for_init(identifier, temporary_directory, corpus, dictionary)
        done_path = os.path.join(self.train_directory, 'done')
        dirty_path = os.path.join(self.train_directory, 'dirty')
        self.feature_config.fmllr = True
        if os.path.exists(done_path):
            self.logger.info('{} training already done, skipping initialization.'.format(self.identifier))
            return
        begin = time.time()
        if os.path.exists(os.path.join(self.train_directory, '1.mdl')):
            return

        self.logger.info('Initializing speaker-adapted triphone training...')
        align_directory = previous_trainer.align_directory
        context_opts = []
        ci_phones = self.dictionary.silence_csl
        try:
            if os.path.exists(os.path.join(align_directory, 'lda.mat')):
                shutil.copyfile(os.path.join(align_directory, 'lda.mat'), os.path.join(self.train_directory, 'lda.mat'))
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

            with open(extra_question_int_path, 'r') as in_file, \
                    open(questions_path, 'a') as out_file:
                for line in in_file:
                    out_file.write(line)

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
            os.remove(treeacc_path)

            compile_train_graphs(self.train_directory, self.dictionary.output_directory,
                                 self.data_directory, self.corpus.num_jobs, self)
            os.rename(occs_path, os.path.join(self.train_directory, '1.occs'))
            os.rename(mdl_path, os.path.join(self.train_directory, '1.mdl'))

            convert_alignments(self.train_directory, align_directory, self.corpus.num_jobs, self)

            if os.path.exists(os.path.join(align_directory, 'trans.0')):
                for i in range(self.corpus.num_jobs):
                    shutil.copy(os.path.join(align_directory, 'trans.{}'.format(i)),
                                os.path.join(self.train_directory, 'trans.{}'.format(i)))
            else:

                calc_fmllr(self.train_directory, self.data_directory,
                           self.dictionary.silence_csl, self.corpus.num_jobs, self, initial=True)
            parse_logs(self.log_directory)
        except Exception as e:
            with open(dirty_path, 'w'):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise
        self.logger.info('Initialization complete!')
        self.logger.debug('Initialization took {} seconds'.format(time.time() - begin))
