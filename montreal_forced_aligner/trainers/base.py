from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Any, Optional, Union, Callable, List
if TYPE_CHECKING:
    from ..config import ConfigDict
    from ..corpus import AlignableCorpus, CorpusType
    from ..dictionary import DictionaryType
    from ..trainers import BaseTrainer
    from ..aligner import PretrainedAligner
    from ..models import MetaDict
    TrainerType = Union[BaseTrainer, PretrainedAligner]
import os
import re
import time
from tqdm import tqdm
import subprocess
import shutil

from .. import __version__
from ..exceptions import TrainerError, KaldiProcessingError
from ..utils import thirdparty_binary, log_kaldi_errors, parse_logs

from ..multiprocessing import (align, acc_stats, convert_ali_to_textgrids, compile_information,
                               compute_alignment_improvement, compile_train_graphs)

from ..models import AcousticModel
from ..features.config import FeatureConfig


class BaseTrainer(object):
    """
    Configuration class for all trainings


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
    """

    def __init__(self, default_feature_config: FeatureConfig):
        self.logger = None
        self.corpus = None
        self.dictionary = None
        self.transition_scale = 1.0
        self.acoustic_scale = 0.1
        self.self_loop_scale = 0.1
        self.realignment_iterations = []
        self.num_iterations = 40
        self.beam = 10
        self.retry_beam = 40
        self.max_gaussians = 1000
        self.boost_silence = 1.0
        self.power = 0.25
        self.subset = None
        self.calc_pron_probs = False
        self.architecture = 'gmm-hmm'
        self.feature_config = FeatureConfig()
        self.feature_config.update(default_feature_config.params())
        self.initial_gaussians = None  # Gets set later
        self.temp_directory = None
        self.identifier = None
        self.corpus: Optional[AlignableCorpus] = None
        self.data_directory = None
        self.debug = False
        self.use_mp = True
        self.current_gaussians = None
        self.iteration = 0
        self.training_complete = False
        self.speaker_independent = True
        self.uses_cmvn = True
        self.uses_splices = False
        self.uses_voiced = False
        self.previous_trainer: Optional[BaseTrainer] = None

    @property
    def train_directory(self) -> str:
        return os.path.join(self.temp_directory, self.identifier)

    @property
    def log_directory(self) -> str:
        return os.path.join(self.train_directory, 'log')

    @property
    def align_directory(self) -> str:
        return os.path.join(self.temp_directory, f'{self.identifier}_ali')

    @property
    def align_log_directory(self) -> str:
        return os.path.join(self.align_directory, 'log')

    @property
    def working_directory(self) -> str:
        if self.training_complete:
            return self.align_directory
        return self.train_directory

    @property
    def working_log_directory(self) -> str:
        if self.training_complete:
            return self.align_log_directory
        return self.log_directory

    @property
    def fmllr_options(self) -> ConfigDict:
        raise NotImplementedError

    @property
    def lda_options(self) -> ConfigDict:
        raise NotImplementedError

    @property
    def tree_path(self):
        return os.path.join(self.working_directory, 'tree')

    @property
    def current_model_path(self):
        if self.training_complete or self.iteration is None or self.iteration > self.num_iterations:
            return os.path.join(self.working_directory, f'final.mdl')
        return os.path.join(self.working_directory, f'{self.iteration}.mdl')

    @property
    def next_model_path(self):
        if self.iteration > self.num_iterations:
            return os.path.join(self.working_directory, f'final.mdl')
        return os.path.join(self.working_directory, f'{self.iteration + 1}.mdl')

    @property
    def next_occs_path(self):
        if self.training_complete:
            return os.path.join(self.working_directory, f'final.occs')
        return os.path.join(self.working_directory, f'{self.iteration + 1}.occs')

    @property
    def alignment_model_path(self):
        path = os.path.join(self.working_directory, f'final.alimdl')
        if self.speaker_independent and os.path.exists(path):
            return path
        if not self.training_complete:
            return self.current_model_path
        return os.path.join(self.working_directory, f'final.mdl')


    def compute_calculated_properties(self) -> None:
        pass

    @property
    def train_type(self) -> str:
        raise NotImplementedError

    @property
    def phone_type(self) -> str:
        raise NotImplementedError

    @property
    def final_gaussian_iteration(self) -> int:
        return self.num_iterations - 10

    @property
    def gaussian_increment(self) -> int:
        return int((self.max_gaussians - self.initial_gaussians) / self.final_gaussian_iteration)

    @property
    def align_options(self) -> ConfigDict:
        options_silence_csl = ''
        if self.dictionary:
            options_silence_csl = self.dictionary.optional_silence_csl
        return {'beam': self.beam, 'retry_beam': self.retry_beam, 'transition_scale': self.transition_scale,
                'acoustic_scale': self.acoustic_scale, 'self_loop_scale': self.self_loop_scale,
                'boost_silence': self.boost_silence, 'debug': self.debug,
                'optional_silence_csl': options_silence_csl}

    def analyze_align_stats(self) -> None:
        unaligned, log_like = compile_information(self)

        self.logger.debug(f'Average per frame likelihood (this might not actually mean anything) '
                          f'for {self.identifier}: {log_like}')
        self.logger.debug(f'Number of unaligned files '
                          f'for {self.identifier}: {len(unaligned)}')

    def update(self, data: Dict[str, Any]) -> None:
        from ..config.base_config import PARSING_KEYS
        for k, v in data.items():
            if k == 'use_mp':
                self.feature_config.use_mp = v
            if k == 'features':
                self.feature_config.update(v)
            elif k in PARSING_KEYS:
                continue
            elif not hasattr(self, k):
                raise TrainerError(f'No field found for key {k}')
            else:
                setattr(self, k, v)
        self.compute_calculated_properties()

    def _setup_for_init(self, identifier: str, temporary_directory: str, corpus: AlignableCorpus,
                        dictionary: DictionaryType, previous_trainer: Optional[TrainerType]) -> None:
        begin = time.time()
        self.temp_directory = temporary_directory
        self.identifier = identifier
        dirty_path = os.path.join(self.train_directory, 'dirty')
        done_path = os.path.join(self.align_directory, 'done')
        if os.path.exists(dirty_path):  # if there was an error, let's redo from scratch
            shutil.rmtree(self.train_directory)
        self.logger.info(f'Initializing training for {identifier}...')
        self.corpus = corpus
        self.dictionary = dictionary
        self.previous_trainer = previous_trainer
        if os.path.exists(done_path):
            self.training_complete = True
            self.iteration = None
            return
        os.makedirs(self.train_directory, exist_ok=True)
        os.makedirs(self.align_directory, exist_ok=True)
        os.makedirs(self.log_directory, exist_ok=True)
        os.makedirs(self.align_log_directory, exist_ok=True)
        if self.subset is not None and self.subset > corpus.num_utterances:
            self.logger.warning('Subset specified is larger than the dataset, '
                                'using full corpus for this training block.')

        try:
            self.data_directory = corpus.split_directory
            self.corpus.generate_features()
            if self.subset is not None:
                self.data_directory = corpus.subset_directory(self.subset)
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise
        self.logger.debug(f'Setup for initialization took {time.time() - begin} seconds')

    def increment_gaussians(self):
        self.current_gaussians += self.gaussian_increment

    def init_training(self, identifier: str, temporary_directory: str, corpus: AlignableCorpus, dictionary: DictionaryType,
                      previous_trainer: Optional[TrainerType]) -> None:
        raise NotImplementedError

    def get_unaligned_utterances(self) -> List[str]:
        error_regex = re.compile(r'Did not successfully decode file (\w+),')
        error_files = []
        for j in self.corpus.jobs:
            path = os.path.join(self.align_directory, 'log', f'align.{j.name}.log')
            if not os.path.exists(path):
                continue
            with open(path, 'r') as f:
                error_files.extend(error_regex.findall(f.read()))
        return error_files

    def align(self, subset: Optional[int]=None) -> None:
        dirty_path = os.path.join(self.align_directory, 'dirty')
        if os.path.exists(dirty_path):  # if there was an error, let's redo from scratch
            shutil.rmtree(self.align_directory)
        done_path = os.path.join(self.align_directory, 'done')
        if not os.path.exists(done_path):
            message = f'Generating alignments using {self.identifier} models'
            if subset:
                message += f' using {subset} utterances...'
            else:
                message += ' for the whole corpus...'
            self.logger.info(message)
            begin = time.time()
            if subset is None:
                self.data_directory = self.corpus.split_directory
            else:
                self.data_directory = self.corpus.subset_directory(subset)
            try:
                self.iteration = None
                compile_train_graphs(self)
                align(self)
                self.analyze_align_stats()
                self.save(os.path.join(self.align_directory, 'acoustic_model.zip'))
            except Exception as e:
                with open(dirty_path, 'w'):
                    pass
                if isinstance(e, KaldiProcessingError):
                    log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
                raise
            with open(done_path, 'w'):
                pass
            self.logger.debug(f'Alignment took {time.time() - begin} seconds')
        else:
            self.logger.info(f'Alignments using {self.identifier} models already done')

    def training_iteration(self):
        if os.path.exists(self.next_model_path):
            self.iteration += 1
            return
        if self.iteration in self.realignment_iterations:
            align(self)
            if self.debug:
                compute_alignment_improvement(self)
        acc_stats(self)

        parse_logs(self.log_directory)
        if self.iteration < self.final_gaussian_iteration:
            self.increment_gaussians()
        self.iteration += 1

    def train(self):
        done_path = os.path.join(self.train_directory, 'done')
        dirty_path = os.path.join(self.train_directory, 'dirty')
        if os.path.exists(done_path):
            self.logger.info(f'{self.identifier} training already done, skipping initialization.')
            return
        begin = time.time()
        try:
            with tqdm(total=self.num_iterations) as pbar:
                while self.iteration < self.num_iterations + 1:
                    self.training_iteration()
                    pbar.update(1)
            self.finalize_training()
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
        self.logger.debug(f'Training took {time.time() - begin} seconds')

    def finalize_training(self):
        shutil.copy(os.path.join(self.train_directory, f'{self.num_iterations}.mdl'),
                    self.next_model_path)
        shutil.copy(os.path.join(self.train_directory, f'{self.num_iterations}.occs'),
                    os.path.join(self.train_directory, 'final.occs'))
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
        if not self.debug:
            for i in range(1, self.num_iterations):
                model_path = os.path.join(self.train_directory, f'{i}.mdl')
                try:
                    os.remove(model_path)
                except FileNotFoundError:
                    pass
                try:
                    os.remove(os.path.join(self.train_directory, f'{i}.occs'))
                except FileNotFoundError:
                    pass
        self.training_complete = True
        self.iteration = None

    @property
    def meta(self) -> MetaDict:
        from datetime import datetime
        data = {'phones': sorted(self.dictionary.nonsil_phones),
                'version': __version__,
                'architecture': self.architecture,
                'train_date': str(datetime.now()),
                'features': self.feature_config.params(),
                'multilingual_ipa': self.dictionary.multilingual_ipa
                }
        if self.dictionary.multilingual_ipa:
            data['strip_diacritics'] = self.dictionary.strip_diacritics
            data['digraphs'] = self.dictionary.digraphs
        return data

    def export_textgrids(self) -> None:
        """
        Export a TextGrid file for every sound file in the dataset
        """
        begin = time.time()
        try:
            convert_ali_to_textgrids(self)
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise
        self.logger.debug(f'Exporting textgrids took {time.time() - begin} seconds')

    def save(self, path: str, root_directory: Optional[str]=None) -> None:
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
        acoustic_model = AcousticModel.empty(basename, root_directory=root_directory)
        acoustic_model.add_meta_file(self)
        acoustic_model.add_model(self.train_directory)
        if directory:
            os.makedirs(directory, exist_ok=True)
        basename, _ = os.path.splitext(path)
        acoustic_model.dump(path)
