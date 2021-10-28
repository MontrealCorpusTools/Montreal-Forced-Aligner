from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable
if TYPE_CHECKING:
    from ..corpus import AlignableCorpus
    from .pretrained import PretrainedAligner
    from ..dictionary import Dictionary
    from ..config import AlignConfig
    from logging import Logger
    from ..models import MetaDict

import os
import shutil

from .. import __version__
from .base import BaseAligner
from ..multiprocessing import (align, compile_train_graphs, train_map,
                                         calc_fmllr, compile_information)
from ..exceptions import KaldiProcessingError
from ..utils import log_kaldi_errors
from ..models import AcousticModel

class AdaptingAligner(BaseAligner):
    def __init__(self, corpus: AlignableCorpus, dictionary: Dictionary, previous_aligner: PretrainedAligner,
                 align_config: AlignConfig,
                 temp_directory: Optional[str] = None,
                 call_back: Optional[Callable] = None, debug: bool = False, verbose: bool = False,
                 logger: Optional[Logger] = None):
        self.previous_aligner = previous_aligner
        self.acoustic_model = self.previous_aligner.acoustic_model
        super().__init__(corpus, dictionary, align_config, temp_directory,
                                                call_back, debug, verbose, logger)
        self.align_config.data_directory = corpus.split_directory
        log_dir = os.path.join(self.align_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)
        self.align_config.logger = self.logger
        self.logger.info('Done with setup!')
        self.training_complete = False
        self.mapping_tau = 20

    def setup(self) -> None:
        super().setup()
        self.previous_aligner.align()
        self.acoustic_model.export_model(self.align_directory)
        for f in ['final.mdl', 'final.alimdl']:
            p = os.path.join(self.align_directory, f)
            if not os.path.exists(p):
                continue
            os.rename(p, os.path.join(self.align_directory, f.replace('final', '0')))

    @property
    def align_directory(self) -> str:
        return os.path.join(self.temp_directory, 'adapt')

    @property
    def current_model_path(self):
        if self.training_complete:
            return os.path.join(self.working_directory, f'final.mdl')
        return os.path.join(self.working_directory, f'0.mdl')

    @property
    def next_model_path(self):
        return os.path.join(self.working_directory, f'final.mdl')


    def adapt(self) -> None:
        done_path = os.path.join(self.align_directory, 'done')
        dirty_path = os.path.join(self.align_directory, 'dirty')
        if os.path.exists(done_path):
            self.logger.info('Adapting already done, skipping.')
            return
        try:
            self.logger.info('Adapting pretrained model...')
            train_map(self)
            self.training_complete = True
        except Exception as e:
            with open(dirty_path, 'w'):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise
        with open(done_path, 'w'):
            pass

    @property
    def meta(self) -> MetaDict:
        from datetime import datetime
        data = {'phones': sorted(self.dictionary.nonsil_phones),
                'version': __version__,
                'architecture': self.acoustic_model.meta['architecture'],
                'train_date': str(datetime.now()),
                'features': self.previous_aligner.align_config.feature_config.params(),
                'multilingual_ipa': self.dictionary.multilingual_ipa
                }
        if self.dictionary.multilingual_ipa:
            data['strip_diacritics'] = self.dictionary.strip_diacritics
            data['digraphs'] = self.dictionary.digraphs
        return data

    def save(self, path, root_directory=None) -> None:
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
        acoustic_model.add_model(self.align_directory)
        if directory:
            os.makedirs(directory, exist_ok=True)
        basename, _ = os.path.splitext(path)
        acoustic_model.dump(path)

    def align(self, subset: Optional[int]=None) -> None:
        done_path = os.path.join(self.align_directory, 'done')
        dirty_path = os.path.join(self.align_directory, 'dirty')
        if os.path.exists(done_path):
            self.logger.info('Alignment already done, skipping.')
            return
        try:
            compile_train_graphs(self)
            log_dir = os.path.join(self.align_directory, 'log')
            os.makedirs(log_dir, exist_ok=True)

            self.logger.info('Performing first-pass alignment...')
            self.speaker_independent = True
            align(self)
            unaligned, average_log_like = compile_information(self)
            self.logger.debug(f'Prior to SAT, average per frame likelihood (this might not actually mean anything): {average_log_like}')
            if not self.align_config.disable_sat and self.previous_aligner.acoustic_model.feature_config.fmllr \
                    and not os.path.exists(os.path.join(self.align_directory, 'trans.0')):
                self.logger.info('Calculating fMLLR for speaker adaptation...')
                calc_fmllr(self)

                self.speaker_independent = False
                self.logger.info('Performing second-pass alignment...')
                align(self)

                unaligned, average_log_like = compile_information(self)
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