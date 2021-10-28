from __future__ import annotations
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from ..config import FeatureConfig
    from ..models import MetaDict
    from ..corpus import AlignableCorpus
    from ..dictionary import DictionaryType
    from ..aligner import PretrainedAligner
from typing import Dict, Any
import os
from tqdm import tqdm
import subprocess
import shutil
import time

from .base import BaseTrainer
from ..utils import thirdparty_binary, log_kaldi_errors, parse_logs
from ..helper import load_scp
from ..exceptions import KaldiProcessingError

from ..multiprocessing.ivector import (gmm_gselect, acc_global_stats, gauss_to_post,
                                           acc_ivector_stats, extract_ivectors)

from ..models import IvectorExtractor

IvectorConfigType = Dict[str, Any]


class IvectorExtractorTrainer(BaseTrainer):
    """
    Configuration class for job_name-vector extractor training

    Attributes
    ----------
    ivector_dimension : int
        Dimension of the extracted job_name-vector
    ivector_period : int
        Number of frames between job_name-vector extractions
    num_iterations : int
        Number of training iterations to perform
    num_gselect : int
        Gaussian-selection using diagonal model: number of Gaussians to select
    posterior_scale : float
        Scale on the acoustic posteriors, intended to account for inter-frame correlations
    min_post : float
        Minimum posterior to use (posteriors below this are pruned out)
    subsample : int
        Speeds up training; training on every job_name'th feature
    max_count : int
        The use of this option (e.g. --max-count 100) can make iVectors more consistent for different lengths of utterance, by scaling up the prior term when the data-count exceeds this value. The data-count is after posterior-scaling, so assuming the posterior-scale is 0.1, --max-count 100 starts having effect after 1000 frames, or 10 seconds of data.
    """

    def __init__(self, default_feature_config: FeatureConfig):
        super(IvectorExtractorTrainer, self).__init__(default_feature_config)

        self.ubm_num_iterations = 4
        self.ubm_num_gselect = 30
        self.ubm_num_frames = 500000
        self.ubm_num_gaussians = 256
        self.ubm_num_iterations_init = 20
        self.ubm_initial_gaussian_proportion = 0.5
        self.ubm_min_gaussian_weight = 0.0001

        self.ubm_remove_low_count_gaussians = True

        self.ivector_dimension = 128
        self.num_iterations = 10
        self.num_gselect = 20
        self.posterior_scale = 1.0
        self.silence_weight = 0.0
        self.min_post = 0.025
        self.gaussian_min_count = 100
        self.subsample = 5
        self.max_count = 100
        self.apply_cmn = True
        self.previous_align_directory = None
        self.dubm_training_complete = False
        self.ubm_training_complete = False

    @property
    def meta(self) -> MetaDict:
        from .. import __version__
        return {
            'version': __version__,
            'ivector_dimension': self.ivector_dimension,
            'apply_cmn': self.apply_cmn,
            'num_gselect': self.num_gselect,
            'min_post': self.min_post,
            'posterior_scale': self.posterior_scale,
            'features': self.feature_config.params(),
        }

    @property
    def train_type(self) -> str:
        return 'ivector'

    @property
    def align_directory(self) -> str:
        return self.train_directory

    @property
    def ivector_options(self) -> MetaDict:
        return {'subsample': self.subsample, 'num_gselect': self.num_gselect, 'posterior_scale': self.posterior_scale,
                'min_post': self.min_post, 'silence_weight': self.silence_weight, 'max_count': self.max_count,
                'ivector_dimension': self.ivector_dimension,
                'sil_phones': self.dictionary.silence_csl
                }

    @property
    def current_ie_path(self):
        if self.training_complete or self.iteration is None or self.iteration > self.num_iterations:
            return os.path.join(self.working_directory, f'final.ie')
        return os.path.join(self.working_directory, f'{self.iteration}.ie')

    @property
    def next_ie_path(self):
        if self.iteration > self.num_iterations:
            return os.path.join(self.working_directory, f'final.ie')
        return os.path.join(self.working_directory, f'{self.iteration + 1}.ie')

    @property
    def dubm_path(self):
        return os.path.join(self.working_directory, 'final.dubm')

    @property
    def current_dubm_path(self):
        if self.dubm_training_complete:
            return os.path.join(self.working_directory, 'final.dubm')
        return os.path.join(self.working_directory, f'{self.iteration}.dubm')

    @property
    def next_dubm_path(self):
        if self.dubm_training_complete:
            return os.path.join(self.working_directory, 'final.dubm')
        return os.path.join(self.working_directory, f'{self.iteration + 1}.dubm')

    @property
    def current_ubm_path(self):
        if self.ubm_training_complete:
            return os.path.join(self.working_directory, 'final.ubm')
        return os.path.join(self.working_directory, f'{self.iteration}.ubm')

    @property
    def ie_path(self):
        return os.path.join(self.working_directory, 'final.ie')

    @property
    def model_path(self):
        return os.path.join(self.working_directory, 'final.mdl')

    @property
    def next_ubm_path(self):
        if self.ubm_training_complete:
            return os.path.join(self.working_directory, 'final.ubm')
        return os.path.join(self.working_directory, f'{self.iteration + 1}.ubm')

    def train_ubm_iteration(self):
        # Accumulate stats
        acc_global_stats(self)
        self.iteration += 1

    def finalize_train_ubm(self):
        final_dubm_path = os.path.join(self.train_directory, 'final.dubm')
        shutil.copy(os.path.join(self.train_directory, f'{self.ubm_num_iterations}.dubm'),
                    final_dubm_path)
        self.iteration = 0
        self.dubm_training_complete = True

    def train_ubm(self) -> None:
        # train diag ubm
        final_ubm_path = os.path.join(self.train_directory, 'final.ubm')
        if os.path.exists(final_ubm_path):
            return
        begin = time.time()
        self.logger.info('Initializing diagonal UBM...')
        # Initialize model from E-M in memory
        log_directory = os.path.join(self.train_directory, 'log')
        num_gauss_init = int(self.ubm_initial_gaussian_proportion * int(self.ubm_num_gaussians))
        log_path = os.path.join(log_directory, 'gmm_init.log')
        feat_name = self.feature_config.feature_id
        all_feats_path = os.path.join(self.corpus.output_directory, f'{feat_name}.scp')
        feature_string = self.corpus.jobs[0].construct_base_feature_string(self.corpus, all_feats=True
                                                                            )
        with open(all_feats_path, 'w') as outf:
            for i in self.corpus.jobs:
                feat_paths = i.construct_path_dictionary(self.data_directory, 'feats', 'scp')
                for p in feat_paths.values():
                    with open(p) as inf:
                        for line in inf:
                            outf.write(line)
        self.iteration = 0
        with open(log_path, 'w') as log_file:
            gmm_init_proc = subprocess.Popen([thirdparty_binary('gmm-global-init-from-feats'),
                                              f'--num-threads={self.corpus.num_jobs}',
                                              f'--num-frames={self.ubm_num_frames}',
                                              f'--num_gauss={self.ubm_num_gaussians}',
                                              f'--num_gauss_init={num_gauss_init}',
                                              f'--num_iters={self.ubm_num_iterations_init}',
                                              feature_string,
                                              self.current_dubm_path],
                                             stderr=log_file)
            gmm_init_proc.communicate()
        # Store Gaussian selection indices on disk
        gmm_gselect(self)
        final_dubm_path = os.path.join(self.train_directory, 'final.dubm')

        if not os.path.exists(final_dubm_path):
            self.logger.info('Training diagonal UBM...')
            with tqdm(total=self.ubm_num_iterations) as pbar:
                while self.iteration < self.ubm_num_iterations + 1:
                    self.train_ubm_iteration()
                    pbar.update(1)
        self.finalize_train_ubm()
        parse_logs(log_directory)
        self.logger.info('Finished training UBM!')
        self.logger.debug(f'UBM training took {time.time() - begin} seconds')

    def init_training(self, identifier: str, temporary_directory: str, corpus: AlignableCorpus, dictionary: DictionaryType,
                      previous_trainer: Optional[PretrainedAligner]=None) -> None:
        self._setup_for_init(identifier, temporary_directory, corpus, dictionary, previous_trainer)
        done_path = os.path.join(self.train_directory, 'done')
        if os.path.exists(done_path):
            self.logger.info(f'{self.identifier} training already done, skipping initialization.')
            return
        shutil.copyfile(previous_trainer.current_model_path,
                        os.path.join(self.train_directory, 'final.mdl'))
        for p in previous_trainer.ali_paths:
            shutil.copyfile(p,
                            p.replace(previous_trainer.working_directory, self.train_directory))
        self.corpus.write_utt2spk()
        begin = time.time()
        self.previous_align_directory = previous_trainer.align_directory

        self.train_ubm()
        self.init_ivector_train()
        self.logger.info('Initialization complete!')
        self.logger.debug(f'Initialization took {time.time() - begin} seconds')

    def init_ivector_train(self) -> None:
        init_ie_path = os.path.join(self.train_directory, '0.ie')
        if os.path.exists(init_ie_path):
            return
        self.iteration = 0
        begin = time.time()
        # Initialize job_name-vector extractor
        log_directory = os.path.join(self.train_directory, 'log')
        log_path = os.path.join(log_directory, 'init.log')
        diag_ubm_path = os.path.join(self.train_directory, 'final.dubm')
        full_ubm_path = os.path.join(self.train_directory, 'final.ubm')
        with open(log_path, 'w') as log_file:
            subprocess.call([thirdparty_binary('gmm-global-to-fgmm'),
                             diag_ubm_path,
                             full_ubm_path],
                            stderr=log_file)
            subprocess.call([thirdparty_binary('ivector-extractor-init'),
                             f'--ivector-dim={self.ivector_dimension}',
                             '--use-weights=false',
                             full_ubm_path,
                             self.current_ie_path],
                            stderr=log_file)

        # Do Gaussian selection and posterior extraction
        gauss_to_post(self)
        parse_logs(log_directory)
        self.logger.debug(f'Initialization ivectors took {time.time() - begin} seconds')

    def align(self, subset: Optional[int]=None):
        self.save(os.path.join(self.align_directory, 'ivector_extractor.zip'))

    def training_iteration(self):
        if os.path.exists(self.next_ie_path):
            self.iteration += 1
            return
        # Accumulate stats and sum
        acc_ivector_stats(self)


        self.iteration += 1

    def finalize_training(self):
        from sklearn.naive_bayes import GaussianNB
        from joblib import dump
        import numpy as np
        # Rename to final
        shutil.copy(os.path.join(self.train_directory, f'{self.num_iterations}.ie'),
                    os.path.join(self.train_directory, 'final.ie'))
        self.training_complete = True
        self.iteration = None
        extract_ivectors(self)
        x = []
        y = []
        speakers = sorted(self.corpus.speakers.keys())
        for j in self.corpus.jobs:
            arguments = j.extract_ivector_arguments(self)
            for ivector_path in arguments.ivector_paths.values():
                ivec = load_scp(ivector_path)
                for utt, ivector in ivec.items():
                    ivector = [float(x) for x in ivector]
                    s = self.corpus.utterances[utt].speaker.name
                    s_ind = speakers.index(s)
                    y.append(s_ind)
                    x.append(ivector)
        x = np.array(x)
        y = np.array(y)
        clf = GaussianNB()
        clf.fit(x, y)
        clf_param_path = os.path.join(self.train_directory, 'speaker_classifier.mdl')
        dump(clf, clf_param_path)
        classes_path = os.path.join(self.train_directory, 'speaker_labels.txt')
        with open(classes_path, 'w', encoding='utf8') as f:
            for i, s in enumerate(speakers):
                f.write(f'{s} {i}\n')

    def save(self, path: str, root_directory: Optional[str]=None):
        """
        Output an acoustic model and dictionary to the specified path

        Parameters
        ----------
        path : str
            Path to save acoustic model and dictionary
        root_directory : str or None
            Path for root directory of temporary files
        """
        directory, filename = os.path.split(path)
        basename, _ = os.path.splitext(filename)
        ivector_extractor = IvectorExtractor.empty(basename, root_directory)
        ivector_extractor.add_meta_file(self)
        ivector_extractor.add_model(self.train_directory)
        os.makedirs(directory, exist_ok=True)
        basename, _ = os.path.splitext(path)
        ivector_extractor.dump(basename)
