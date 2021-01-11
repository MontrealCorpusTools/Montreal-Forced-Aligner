import os
from tqdm import tqdm
import subprocess
import shutil

from .base import BaseTrainer
from ..helper import thirdparty_binary, make_path_safe, log_kaldi_errors, parse_logs
from ..exceptions import KaldiProcessingError

from ..multiprocessing import (gmm_gselect, acc_global_stats, gauss_to_post, acc_ivector_stats, extract_ivectors)

from ..models import IvectorExtractor


class IvectorExtractorTrainer(BaseTrainer):
    """
    Configuration class for i-vector extractor training

    Attributes
    ----------
    ivector_dimension : int
        Dimension of the extracted i-vector
    ivector_period : int
        Number of frames between i-vector extractions
    num_iterations : int
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
        self.ubm_num_frames = 500000
        self.ubm_num_gaussians = 256
        self.num_components = 2048
        self.ubm_num_iterations_init = 20
        self.ubm_initial_gaussian_proportion = 0.5
        self.ubm_cleanup = True
        self.ubm_min_gaussian_weight = 0.0001

        self.ubm_remove_low_count_gaussians = True

        self.ivector_dimension = 128
        self.use_weights = False
        self.ivector_period = 10
        self.num_iterations = 10
        self.num_gselect = 20
        self.posterior_scale = 1.0
        self.silence_weight = 0.0
        self.min_post = 0.025
        self.num_samples_for_weights = 3
        self.gaussian_min_count = 100
        self.subsample = 5
        self.max_count = 100
        self.apply_cmn = True

    @property
    def meta(self):
        from .. import __version__
        return {
            'version': __version__,
            'ivector_dimension': self.ivector_dimension,
            'apply_cmn': self.apply_cmn,
            'ivector_period': self.ivector_period,
            'num_gselect': self.num_gselect,
            'min_post': self.min_post,
            'posterior_scale': self.posterior_scale,
            'features': self.feature_config.params(),
        }

    @property
    def train_type(self):
        return 'ivector'

    @property
    def ivector_options(self):
        return {'subsample': self.subsample, 'num_gselect': self.num_gselect, 'posterior_scale': self.posterior_scale,
                'min_post': self.min_post, 'silence_weight': self.silence_weight, 'max_count': self.max_count,
                'ivector_dimension': self.ivector_dimension
                }

    def train_ubm(self, call_back=None):
        if call_back is None:
            call_back = print
        # train diag ubm
        self.logger.info('Initializing diagonal UBM...')
        # Initialize model from E-M in memory
        log_directory = os.path.join(self.train_directory, 'log')
        num_gauss_init = int(self.ubm_initial_gaussian_proportion * int(self.ubm_num_gaussians))
        log_path = os.path.join(log_directory, 'gmm_init.log')

        all_feats_path = os.path.join(self.corpus.output_directory, self.feature_config.voiced_feature_id + '.scp')
        with open(all_feats_path, 'w') as outf:
            for i in range(self.corpus.num_jobs):
                with open(os.path.join(self.data_directory,
                                       self.feature_config.voiced_feature_id + '.{}.scp'.format(i))) as inf:
                    for line in inf:
                        outf.write(line)
        with open(log_path, 'w') as log_file:
            gmm_init_proc = subprocess.Popen([thirdparty_binary('gmm-global-init-from-feats'),
                                              '--num-threads={}'.format(self.corpus.num_jobs),
                                              '--num-frames={}'.format(self.ubm_num_frames),
                                              '--num_gauss={}'.format(self.ubm_num_gaussians),
                                              '--num_gauss_init={}'.format(num_gauss_init),
                                              '--num_iters={}'.format(self.ubm_num_iterations_init),
                                              'scp:' + all_feats_path,
                                              os.path.join(self.train_directory, '0.dubm')],
                                             stderr=log_file)
            gmm_init_proc.communicate()

        # Store Gaussian selection indices on disk
        gmm_gselect('0', self, self.corpus.num_jobs)

        self.logger.info('Training diagonal UBM...')
        if call_back == print:
            iters = tqdm(range(0, self.ubm_num_iterations))
        else:
            iters = range(0, self.ubm_num_iterations)
        for i in iters:
            # Accumulate stats
            acc_global_stats(self, self.corpus.num_jobs, i, full=False)

            # Don't remove low-count Gaussians till the last tier,
            # or gselect info won't be valid anymore
            if i < self.ubm_num_iterations - 1:
                opt = '--remove-low-count-gaussians=false'
            else:
                opt = '--remove-low-count-gaussians={}'.format(self.ubm_remove_low_count_gaussians)

            log_path = os.path.join(self.train_directory, 'log', 'update.{}.log'.format(i))
            with open(log_path, 'w') as log_file:
                acc_files = [os.path.join(self.train_directory, '{}.{}.acc'.format(i, x))
                             for x in range(self.corpus.num_jobs)]
                gmm_global_est_proc = subprocess.Popen([thirdparty_binary('gmm-global-est'),
                                                        opt,
                                                        '--min-gaussian-weight=' + str(self.ubm_min_gaussian_weight),
                                                        os.path.join(self.train_directory, '{}.dubm'.format(i)),
                                                        "{} - {}|".format(thirdparty_binary('gmm-global-sum-accs'),
                                                                          ' '.join(map(make_path_safe, acc_files))),
                                                        os.path.join(self.train_directory, '{}.dubm'.format(i + 1))],
                                                       stderr=log_file)
                gmm_global_est_proc.communicate()

            # Move files
        shutil.copy(os.path.join(self.train_directory, '{}.dubm'.format(self.ubm_num_iterations)),
                    os.path.join(self.train_directory, 'final.dubm'))

        # train full ubm

        # Convert final.ubm to fgmm
        log_path = os.path.join(self.train_directory, 'log', 'global_to_fgmm.log')
        with open(log_path, 'w') as log_file:
            subprocess.call([thirdparty_binary('gmm-global-to-fgmm'),
                             os.path.join(self.train_directory, 'final.dubm'),
                             os.path.join(self.train_directory, '0.ubm')],
                            stdout=subprocess.PIPE,
                            stderr=log_file)
        gmm_gselect('final', self, self.corpus.num_jobs)

        self.logger.info('Training full UBM...')
        if call_back == print:
            iters = tqdm(range(0, self.ubm_num_iterations))
        else:
            iters = range(0, self.ubm_num_iterations)
        for i in iters:
            # Accumulate stats
            acc_global_stats(self, self.corpus.num_jobs, i, full=True)

            # Don't remove low-count Gaussians till the last tier,
            # or gselect info won't be valid anymore
            if i < self.ubm_num_iterations - 1:
                opt = '--remove-low-count-gaussians=false'
            else:
                opt = '--remove-low-count-gaussians={}'.format(self.ubm_remove_low_count_gaussians)

            log_path = os.path.join(self.train_directory, 'log', 'update.{}.log'.format(i))
            with open(log_path, 'w') as log_file:
                acc_files = [os.path.join(self.train_directory, '{}.{}.acc'.format(i, x))
                             for x in range(self.corpus.num_jobs)]
                gmm_global_est_proc = subprocess.Popen([thirdparty_binary('fgmm-global-est'),
                                                        opt,
                                                        '--min-gaussian-weight=' + str(self.ubm_min_gaussian_weight),
                                                        os.path.join(self.train_directory, '{}.ubm'.format(i)),
                                                        "{} - {}|".format(thirdparty_binary('fgmm-global-sum-accs'),
                                                                          ' '.join(map(make_path_safe, acc_files))),
                                                        os.path.join(self.train_directory, '{}.ubm'.format(i + 1))],
                                                       stderr=log_file)
                gmm_global_est_proc.communicate()
            # Move files
        shutil.copy(os.path.join(self.train_directory, '{}.ubm'.format(self.ubm_num_iterations)),
                    os.path.join(self.train_directory, 'final.ubm'))
        parse_logs(log_directory)
        self.logger.info('Finished training UBM!')

    def init_training(self, identifier, temporary_directory, corpus, dictionary, previous_trainer):
        self._setup_for_init(identifier, temporary_directory, corpus, dictionary)
        done_path = os.path.join(self.train_directory, 'done')
        dirty_path = os.path.join(self.train_directory, 'dirty')
        if os.path.exists(done_path):
            self.logger.info('{} training already done, skipping initialization.'.format(self.identifier))
            return
        try:
            self.feature_config.compute_vad(self.corpus, logger=self.logger)
            self.feature_config.generate_voiced_features(self.corpus, apply_cmn=self.apply_cmn, logger=self.logger)

            self.train_ubm()
            self.init_ivector_train()
        except Exception as e:
            with open(dirty_path, 'w'):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
            raise

    def init_ivector_train(self):

        # Initialize i-vector extractor
        log_directory = os.path.join(self.train_directory, 'log')
        log_path = os.path.join(log_directory, 'init.log')
        diag_ubm_path = os.path.join(self.train_directory, 'final.dubm')
        full_ubm_path = os.path.join(self.train_directory, 'final.ubm')
        with open(log_path, 'w') as log_file:
            subprocess.call([thirdparty_binary('fgmm-global-to-gmm'),
                             full_ubm_path,
                             diag_ubm_path],
                            stderr=log_file)
            subprocess.call([thirdparty_binary('ivector-extractor-init'),
                             '--ivector-dim=' + str(self.ivector_dimension),
                             '--use-weights={}'.format(self.use_weights).lower(),
                             full_ubm_path,
                             os.path.join(self.train_directory, '0.ie')],
                            stderr=log_file)
        # Do Gaussian selection and posterior extraction
        gmm_gselect('final', self, self.corpus.num_jobs)
        gauss_to_post(self, self.corpus.num_jobs)
        parse_logs(log_directory)
        self.logger.info('Initialization complete!')

    def align(self, subset, call_back=None):
        self.save(os.path.join(self.align_directory, 'ivector_extractor.zip'))

        # extract_ivectors(self, self.corpus.num_jobs)

    def train(self, call_back=None):
        done_path = os.path.join(self.train_directory, 'done')
        dirty_path = os.path.join(self.train_directory, 'dirty')
        if os.path.exists(done_path):
            self.logger.info('{} training already done, skipping training.'.format(self.identifier))
            return
        if call_back == print:
            iters = tqdm(range(0, self.num_iterations))
        else:
            iters = range(0, self.num_iterations)
        try:
            for i in iters:
                # Accumulate stats and sum
                acc_ivector_stats(self, self.corpus.num_jobs, i)

                # Est extractor
                log_path = os.path.join(self.train_directory, 'log', 'update.{}.log'.format(i))
                with open(log_path, 'w') as log_file:
                    extractor_est_proc = subprocess.Popen([thirdparty_binary('ivector-extractor-est'),
                                                           '--num-threads={}'.format(self.corpus.num_jobs),
                                                           '--gaussian-min-count={}'.format(self.gaussian_min_count),
                                                           os.path.join(self.train_directory, '{}.ie'.format(i)),
                                                           os.path.join(self.train_directory, 'acc.{}'.format(i)),
                                                           os.path.join(self.train_directory, '{}.ie'.format(i + 1))],
                                                          stderr=log_file)
                    extractor_est_proc.communicate()
                # Rename to final
            shutil.copy(os.path.join(self.train_directory, '{}.ie'.format(self.num_iterations)),
                        os.path.join(self.train_directory, 'final.ie'))
            os.makedirs(self.corpus.ivector_directory, exist_ok=True)
            shutil.copy(os.path.join(self.train_directory, 'final.ie'),
                        os.path.join(self.corpus.ivector_directory, 'final.ie'))
            shutil.copy(os.path.join(self.train_directory, 'final.dubm'),
                        os.path.join(self.corpus.ivector_directory, 'final.dubm'))
        except Exception as e:
            with open(dirty_path, 'w'):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
            raise
        with open(done_path, 'w'):
            pass

    def save(self, path):
        """
        Output an acoustic model and dictionary to the specified path

        Parameters
        ----------
        path : str
            Path to save acoustic model and dictionary
        """
        directory, filename = os.path.split(path)
        basename, _ = os.path.splitext(filename)
        ivector_extractor = IvectorExtractor.empty(basename)
        ivector_extractor.add_meta_file(self)
        ivector_extractor.add_model(self.train_directory)
        os.makedirs(directory, exist_ok=True)
        basename, _ = os.path.splitext(path)
        ivector_extractor.dump(basename)
