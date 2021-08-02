import os
import re
import subprocess
import time

from .base import BaseTrainer
from ..helper import thirdparty_binary, make_path_safe, log_kaldi_errors, parse_logs
from ..exceptions import KaldiProcessingError

from ..multiprocessing import (mono_align_equal, compile_train_graphs, compute_alignment_improvement)


class MonophoneTrainer(BaseTrainer):
    """
    Configuration class for monophone training


    Attributes
    ----------
    initial_gaussians : int
        Number of gaussians to begin training
    """

    def __init__(self, default_feature_config):
        super(MonophoneTrainer, self).__init__(default_feature_config)
        self.initial_gaussians = 135
        self.compute_calculated_properties()

    def compute_calculated_properties(self):
        for i in range(1, self.num_iterations):
            if i <= int(self.num_iterations / 4):
                self.realignment_iterations.append(i)
            elif i <= int(self.num_iterations * 2 / 4):
                if i - self.realignment_iterations[-1] > 1:
                    self.realignment_iterations.append(i)
            else:
                if i - self.realignment_iterations[-1] > 2:
                    self.realignment_iterations.append(i)

    @property
    def train_type(self):
        return 'mono'

    @property
    def phone_type(self):
        return 'monophone'

    def get_num_gauss(self):
        """
        Get the number of gaussians for a monophone model
        """
        with open(os.devnull, 'w') as devnull:
            proc = subprocess.Popen([thirdparty_binary('gmm-info'),
                                     '--print-args=false',
                                     os.path.join(self.train_directory, '0.mdl')],
                                    stderr=devnull,
                                    stdout=subprocess.PIPE)
            stdout, stderr = proc.communicate()
            num = stdout.decode('utf8')
            matches = re.search(r'gaussians (\d+)', num)
            num = int(matches.groups()[0])
        return num

    def init_training(self, identifier, temporary_directory, corpus, dictionary, previous_trainer=None):
        self._setup_for_init(identifier, temporary_directory, corpus, dictionary)
        done_path = os.path.join(self.train_directory, 'done')
        dirty_path = os.path.join(self.train_directory, 'dirty')
        if os.path.exists(done_path):
            self.logger.info('{} training already done, skipping initialization.'.format(self.identifier))
            return
        begin = time.time()

        tree_path = os.path.join(self.train_directory, 'tree')
        mdl_path = os.path.join(self.train_directory, '0.mdl')

        try:
            feat_dim = corpus.get_feat_dim(self.feature_config)
            feature_string = self.feature_config.construct_feature_proc_string(self.data_directory, self.train_directory, 0)
            #feature_string += " subset-feats --n=10 ark:- ark:-| "
            shared_phones_opt = "--shared-phones=" + os.path.join(dictionary.phones_dir, 'sets.int')
            init_log_path = os.path.join(self.log_directory, 'init.log')
            temp_feats_path = os.path.join(self.train_directory, 'temp_feats')
            with open(init_log_path, 'w') as log_file:
                subprocess.call([thirdparty_binary('subset-feats'), '--n=10',
                                 feature_string, 'ark:'+temp_feats_path], stderr=log_file)
                subprocess.call([thirdparty_binary('gmm-init-mono'), shared_phones_opt,
                                 "--train-feats=ark:"+temp_feats_path,
                                 os.path.join(dictionary.output_directory, 'topo'),
                                 str(feat_dim),
                                 mdl_path,
                                 tree_path],
                                stderr=log_file)
            if os.path.exists(mdl_path):
                os.remove(init_log_path)
            os.remove(temp_feats_path)
            num_gauss = self.get_num_gauss()
            self.initial_gaussians = num_gauss
            compile_train_graphs(self.train_directory, dictionary.output_directory,
                                 self.data_directory, corpus.num_jobs, self)
            mono_align_equal(self.train_directory,
                             self.data_directory, corpus.num_jobs, self)
            log_path = os.path.join(self.train_directory, 'log', 'update.0.log')
            with open(log_path, 'w') as log_file:
                acc_files = [os.path.join(self.train_directory, '0.{}.acc'.format(x)) for x in range(corpus.num_jobs)]
                est_proc = subprocess.Popen([thirdparty_binary('gmm-est'),
                                             '--min-gaussian-occupancy=3',
                                             '--mix-up={}'.format(num_gauss), '--power={}'.format(self.power),
                                             mdl_path, "{} - {}|".format(thirdparty_binary('gmm-sum-accs'),
                                                                         ' '.join(map(make_path_safe, acc_files))),
                                             os.path.join(self.train_directory, '1.mdl')],
                                            stderr=log_file)
                est_proc.communicate()
                if not self.debug:
                    for f in acc_files:
                        os.remove(f)
            parse_logs(self.log_directory)
            if self.debug:
                self.logger.info('Initializing alignment improvement calculations')
                compute_alignment_improvement(0, self, self.train_directory, self.corpus.num_jobs)

        except Exception as e:
            with open(dirty_path, 'w'):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise
        self.logger.info('Initialization complete!')
        self.logger.debug('Initialization took {} seconds'.format(time.time() - begin))
