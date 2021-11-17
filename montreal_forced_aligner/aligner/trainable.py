"""Class definitions for trainable aligners"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..abc import Trainer
from .base import BaseAligner

if TYPE_CHECKING:
    from logging import Logger

    from ..aligner.pretrained import PretrainedAligner
    from ..config import AlignConfig, TrainingConfig
    from ..corpus import Corpus
    from ..dictionary import MultispeakerDictionary

__all__ = ["TrainableAligner"]


class TrainableAligner(BaseAligner, Trainer):
    """
    Aligner that aligns and trains acoustics models on a large dataset

    Parameters
    ----------
    corpus : :class:`~montreal_forced_aligner.corpus.Corpus`
        Corpus object for the dataset
    dictionary : :class:`~montreal_forced_aligner.dictionary.MultispeakerDictionary`
        Dictionary object for the pronunciation dictionary
    training_config : :class:`~montreal_forced_aligner.config.TrainingConfig`
        Configuration to train a model
    align_config : :class:`~montreal_forced_aligner.config.AlignConfig`
        Configuration for alignment
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``
    debug: bool
        Flag for debug mode, default is False
    verbose: bool
        Flag for verbose mode, default is False
    logger: :class:`~logging.Logger`
        Logger to use
    pretrained_aligner: :class:`~montreal_forced_aligner.aligner.pretrained.PretrainedAligner`, optional
        Pretrained aligner to use as input to training
    """

    def __init__(
        self,
        corpus: Corpus,
        dictionary: MultispeakerDictionary,
        training_config: TrainingConfig,
        align_config: AlignConfig,
        temp_directory: Optional[str] = None,
        debug: bool = False,
        verbose: bool = False,
        logger: Optional[Logger] = None,
        pretrained_aligner: Optional[PretrainedAligner] = None,
    ):
        self.training_config = training_config
        self.pretrained_aligner = pretrained_aligner
        if self.pretrained_aligner is not None:
            acoustic_model = pretrained_aligner.acoustic_model
        else:
            acoustic_model = None
        super(TrainableAligner, self).__init__(
            corpus,
            dictionary,
            align_config,
            temp_directory,
            debug,
            verbose,
            logger,
            acoustic_model=acoustic_model,
        )
        for trainer in self.training_config.training_configs:
            trainer.logger = self.logger

    def save(self, path: str, root_directory: Optional[str] = None) -> None:
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
        self.logger.info(f"Saved model to {path}")

    @property
    def meta(self) -> dict:
        """Acoustic model parameters"""
        from ..utils import get_mfa_version

        data = {
            "phones": sorted(self.dictionary.config.non_silence_phones),
            "version": get_mfa_version(),
            "architecture": self.training_config.values()[-1].architecture,
            "phone_type": self.training_config.values()[-1].phone_type,
            "features": self.align_config.feature_config.params(),
        }
        return data

    @property
    def model_path(self) -> str:
        return self.training_config.values()[-1].model_path

    def train(self, generate_final_alignments: bool = True) -> None:
        """
        Run through the training configurations to produce a final acoustic model

        Parameters
        ----------
        generate_final_alignments: bool
            Flag for whether final alignments should be generated at the end of training, defaults to True
        """
        previous = self.pretrained_aligner
        for identifier, trainer in self.training_config.items():
            trainer.debug = self.debug
            trainer.logger = self.logger
            if previous is not None:
                previous.align(trainer.subset)
            trainer.init_training(
                identifier, self.temp_directory, self.corpus, self.dictionary, previous
            )
            trainer.train()
            previous = trainer
        if generate_final_alignments:
            previous.align(None)

    @property
    def align_directory(self) -> str:
        """Align directory"""
        return self.training_config.values()[-1].align_directory
