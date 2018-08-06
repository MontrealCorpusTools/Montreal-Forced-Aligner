import os
from tqdm import tqdm
import subprocess
import shutil

from .base import BaseTrainer
from ..helper import thirdparty_binary, make_path_safe

from ..multiprocessing import (gmm_gselect, acc_global_stats, gauss_to_post, acc_ivector_stats, extract_ivectors)

from ..models import IvectorExtractor


class IvectorExtractorTrainer(BaseTrainer):
    """
    Configuration class for i-vector extractor training

    Attributes
    ----------
    ivector_dim : int
        Dimension of the extracted i-vector
    ivector_period : int
        Number of frames between i-vector extractions
    num_iters : int
        Number of training iterations to perform
    num_gselect : int
        Gaussian-selection using diagonal model: number of Gaussians to select
    posterior_scale : float
        Scale on the acoustic posteriors, intended to account for inter-frame correlations
    min_post : float
        Minimum posterior to use (posteriors below this are pruned out)
    subsample : int
        Speeds up training; training on every x'th feature
    max_count : int
        The use of this option (e.g. --max-count 100) can make iVectors more consistent for different lengths of utterance, by scaling up the prior term when the data-count exceeds this value. The data-count is after posterior-scaling, so assuming the posterior-scale is 0.1, --max-count 100 starts having effect after 1000 frames, or 10 seconds of data.
    """

    def __init__(self, default_feature_config):
        super(IvectorExtractorTrainer, self).__init__(default_feature_config)

        self.ubm_num_iterations = 4
        self.ubm_num_gselect = 30
        self.ubm_num_frames = 400000
        self.ubm_num_gaussians = 256

        self.ubm_num_iterations_init = 20
        self.ubm_initial_gaussian_proportion = 0.5
        self.ubm_cleanup = True
        self.ubm_min_gaussian_weight = 0.0001

        self.ubm_remove_low_count_gaussians = True
        self.ubm_num_threads = 32

        self.ivector_dimension = 100
        self.ivector_period = 10
        self.num_iterations = 10
        self.num_gselect = 5
        self.posterior_scale = 0.1
        self.splice_left_context = 3
        self.splice_right_context = 3
        self.min_post = 0.025
        self.gaussian_min_count = 100
        self.subsample = 2
        self.max_count = 0

        self.num_threads = 4
        self.num_processes = 4

        self.compress = False

    @property
    def meta(self):
        return {'ivector_period': self.ivector_period,
                'splice_left_context': self.splice_left_context,
                'splice_right_context': self.splice_right_context,
                'num_gselect': self.num_gselect,
                'min_post': self.min_post,
                'posterior_scale': self.posterior_scale,
                }

    @property
    def train_type(self):
        return 'ivector'

    def init_training(self, identifier, temporary_directory, corpus, dictionary, previous_trainer):
        self._setup_for_init(identifier, temporary_directory, corpus, dictionary)
        for f in os.listdir(previous_trainer.align_directory):
            if os.path.isdir(os.path.join(previous_trainer.align_directory, f)):
                continue
            shutil.copy(os.path.join(previous_trainer.align_directory, f), os.path.join(self.align_directory, f))
        corpus_directory = self.corpus.output_directory
        lda_mat_path = os.path.join(corpus_directory, 'lda.mat')
        shutil.copy(lda_mat_path, os.path.join(self.train_directory, 'lda.mat'))

        # Initialize model from E-M in memory
        num_gauss_init = int(self.ubm_initial_gaussian_proportion * int(self.ubm_num_gaussians))
        log_path = os.path.join(self.train_directory, 'log', 'gmm_init.log')

        utt2spkpath = os.path.join(self.corpus.output_directory, 'utt2spk')
        cmvnpath = os.path.join(self.corpus.output_directory, 'cmvn.scp')
        featspath = os.path.join(self.corpus.output_directory, 'feats.scp')

        all_feats_path = os.path.join(self.corpus.output_directory, self.feature_config.feature_id + '.scp')
        with open(all_feats_path, 'w') as outf:
            for i in range(self.corpus.num_jobs):
                with open(os.path.join(self.data_directory,
                                       self.feature_config.feature_id + '.{}.scp'.format(i))) as inf:
                    for line in inf:
                        outf.write(line)
        with open(log_path, 'w') as logf:

            gmm_init_proc = subprocess.Popen([thirdparty_binary('gmm-global-init-from-feats'),
                                              '--num-threads=' + str(self.ubm_num_threads),
                                              '--num-frames=' + str(self.ubm_num_frames),
                                              '--num_gauss=' + str(self.ubm_num_gaussians),
                                              '--num_gauss_init=' + str(num_gauss_init),
                                              '--num_iters=' + str(self.ubm_num_iterations_init),
                                              'scp:' + all_feats_path,
                                              os.path.join(self.train_directory, '1.dubm')],
                                             stderr=logf)
            gmm_init_proc.communicate()

        # Store Gaussian selection indices on disk
        gmm_gselect(self, self.corpus.num_jobs)

        for i in range(1, self.ubm_num_iterations):
            # Accumulate stats
            acc_global_stats(self, self.corpus.num_jobs, i)

            # Don't remove low-count Gaussians till the last tier,
            # or gselect info won't be valid anymore
            if i < self.ubm_num_iterations - 1:
                opt = '--remove-low-count-gaussians=false'
            else:
                opt = '--remove-low-count-gaussians=' + str(self.ubm_remove_low_count_gaussians)

            log_path = os.path.join(self.train_directory, 'log', 'update.{}.log'.format(i))
            with open(log_path, 'w') as logf:
                acc_files = [os.path.join(self.train_directory, '{}.{}.acc'.format(i, x))
                             for x in range(self.corpus.num_jobs)]
                gmm_global_est_proc = subprocess.Popen([thirdparty_binary('gmm-global-est'),
                                                        opt,
                                                        '--min-gaussian-weight=' + str(self.ubm_min_gaussian_weight),
                                                        os.path.join(self.train_directory, '{}.dubm'.format(i)),
                                                        "{} - {}|".format(thirdparty_binary('gmm-global-sum-accs'),
                                                                          ' '.join(map(make_path_safe, acc_files))),
                                                        os.path.join(self.train_directory, '{}.dubm'.format(i + 1))],
                                                       stderr=logf)
                gmm_global_est_proc.communicate()

            # Move files
        shutil.copy(os.path.join(self.train_directory, '{}.dubm'.format(self.ubm_num_iterations)),
                    os.path.join(self.train_directory, 'final.dubm'))

        # Convert final.ubm to fgmm
        log_path = os.path.join(self.train_directory, 'log', 'global_to_fgmm.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('gmm-global-to-fgmm'),
                             os.path.join(self.train_directory, 'final.dubm'),
                             os.path.join(self.train_directory, '0.fgmm')],
                            stdout=subprocess.PIPE,
                            stderr=logf)

        # Initialize i-vector extractor
        log_path = os.path.join(self.train_directory, 'log', 'init.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('ivector-extractor-init'),
                             '--ivector-dim=' + str(self.ivector_dimension),
                             '--use-weights=false',
                             os.path.join(self.train_directory, '0.fgmm'),
                             os.path.join(self.train_directory, '1.ie')],
                            stderr=logf)
        # Do Gaussian selection and posterior extraction
        gauss_to_post(self, self.corpus.num_jobs)
        print('Initialization complete!')

    def align(self, subset, call_back=None):
        self.save(os.path.join(self.align_directory, 'ivector_extractor.zip'))

        extract_ivectors(self, self.corpus.num_jobs)

    def train(self, call_back=None):
        if call_back == print:
            iters = tqdm(range(1, self.num_iterations))
        else:
            iters = range(1, self.num_iterations)
        for i in iters:
            # Accumulate stats and sum
            acc_ivector_stats(self, self.corpus.num_jobs, i)

            # Est extractor
            log_path = os.path.join(self.train_directory, 'log', 'update.{}.log'.format(i))
            with open(log_path, 'w') as logf:
                extractor_est_proc = subprocess.Popen([thirdparty_binary('ivector-extractor-est'),
                                                       '--gaussian-min-count={}'.format(self.gaussian_min_count),
                                                       os.path.join(self.train_directory, '{}.ie'.format(i)),
                                                       os.path.join(self.train_directory, 'acc.{}'.format(i)),
                                                       os.path.join(self.train_directory, '{}.ie'.format(i + 1))],
                                                      stderr=logf)
                extractor_est_proc.communicate()
            # Rename to final
        shutil.copy(os.path.join(self.train_directory, '{}.ie'.format(self.num_iterations)),
                    os.path.join(self.train_directory, 'final.ie'))
        os.makedirs(self.corpus.ivector_directory, exist_ok=True)
        shutil.copy(os.path.join(self.train_directory, 'final.ie'), os.path.join(self.corpus.ivector_directory, 'final.ie'))
        shutil.copy(os.path.join(self.train_directory, 'final.dubm'), os.path.join(self.corpus.ivector_directory, 'final.dubm'))

    def save(self, path):
        '''
        Output an acoustic model and dictionary to the specified path

        Parameters
        ----------
        path : str
            Path to save acoustic model and dictionary
        '''
        directory, filename = os.path.split(path)
        basename, _ = os.path.splitext(filename)
        ivector_extractor = IvectorExtractor.empty(basename)
        ivector_extractor.add_meta_file(self)
        ivector_extractor.add_model(self.train_directory)
        os.makedirs(directory, exist_ok=True)
        basename, _ = os.path.splitext(path)
        ivector_extractor.dump(basename)
