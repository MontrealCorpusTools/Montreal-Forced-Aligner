from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable, Collection
if TYPE_CHECKING:
    from ..corpus import AlignableCorpus
    from ..dictionary import Dictionary
    from ..config import AlignConfig
    from ..config import TrainingConfig
    from ..aligner.pretrained import PretrainedAligner
    from logging import Logger

from .base import BaseAligner


class TrainableAligner(BaseAligner):
    """
    Aligner that aligns and trains acoustics models on a large dataset

    Parameters
    ----------
    corpus : :class:`~montreal_forced_aligner.corpus.AlignableCorpus`
        Corpus object for the dataset
    dictionary : :class:`~montreal_forced_aligner.dictionary.Dictionary`
        Dictionary object for the pronunciation dictionary
    training_config : :class:`~montreal_forced_aligner.config.TrainingConfig`
        Configuration to train a model
    align_config : :class:`~montreal_forced_aligner.config.AlignConfig`
        Configuration for alignment
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``
    call_back : callable, optional
        Specifies a call back function for alignment
    """

    def __init__(self, corpus: AlignableCorpus, dictionary: Dictionary, training_config: TrainingConfig,
                 align_config: AlignConfig, temp_directory: Optional[str]=None,
                 call_back: Optional[Callable]=None, debug: bool=False, verbose: bool=False,
                 logger: Optional[Logger]=None, pretrained_aligner: Optional[PretrainedAligner]=None):
        self.training_config = training_config
        self.pretrained_aligner = pretrained_aligner
        super(TrainableAligner, self).__init__(corpus, dictionary, align_config, temp_directory,
                                               call_back, debug, verbose, logger)
        for trainer in self.training_config.training_configs:
            trainer.logger = self.logger

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
        self.training_config.values()[-1].save(path, root_directory)
        self.logger.info('Saved model to {}'.format(path))

    @property
    def meta(self) -> dict:
        from .. import __version__
        data = {'phones': sorted(self.dictionary.nonsil_phones),
                'version': __version__,
                'architecture': self.training_config.values()[-1].architecture,
                'phone_type': self.training_config.values()[-1].phone_type,
                'features': self.align_config.feature_config.params(),
                }
        return data

    def train(self, generate_final_alignments: bool=True) -> None:
        previous = self.pretrained_aligner
        for identifier, trainer in self.training_config.items():
            trainer.debug = self.debug
            trainer.logger = self.logger
            if previous is not None:
                previous.align(trainer.subset)
            trainer.init_training(identifier, self.temp_directory, self.corpus, self.dictionary, previous)
            trainer.train()
            previous = trainer
        if generate_final_alignments:
            previous.align(None)

    @property
    def align_directory(self) -> str:
        return self.training_config.values()[-1].align_directory
