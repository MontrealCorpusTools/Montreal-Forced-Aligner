"""Class definitions for adapting acoustic models"""
from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, Optional

from ..abc import Trainer
from ..exceptions import KaldiProcessingError
from ..models import AcousticModel
from ..multiprocessing import (
    align,
    calc_fmllr,
    compile_information,
    compile_train_graphs,
    train_map,
)
from ..utils import log_kaldi_errors
from .base import BaseAligner

if TYPE_CHECKING:
    from logging import Logger

    from ..config import AlignConfig
    from ..corpus import Corpus
    from ..dictionary import MultispeakerDictionary
    from ..models import MetaDict
    from .pretrained import PretrainedAligner


__all__ = ["AdaptingAligner"]


class AdaptingAligner(BaseAligner, Trainer):
    """
    Aligner adapts another acoustic model to the current data

    Parameters
    ----------
    corpus : :class:`~montreal_forced_aligner.corpus.Corpus`
        Corpus object for the dataset
    dictionary : :class:`~montreal_forced_aligner.dictionary.MultispeakerDictionary`
        Dictionary object for the pronunciation dictionary
    pretrained_aligner: :class:`~montreal_forced_aligner.aligner.PretrainedAligner`
        Pretrained aligner to use as input to training
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
    """

    def __init__(
        self,
        corpus: Corpus,
        dictionary: MultispeakerDictionary,
        previous_aligner: PretrainedAligner,
        align_config: AlignConfig,
        temp_directory: Optional[str] = None,
        debug: bool = False,
        verbose: bool = False,
        logger: Optional[Logger] = None,
    ):
        self.previous_aligner = previous_aligner
        super().__init__(
            corpus,
            dictionary,
            align_config,
            temp_directory,
            debug,
            verbose,
            logger,
            acoustic_model=self.previous_aligner.acoustic_model,
        )
        self.align_config.data_directory = corpus.split_directory
        log_dir = os.path.join(self.align_directory, "log")
        os.makedirs(log_dir, exist_ok=True)
        self.align_config.logger = self.logger
        self.logger.info("Done with setup!")
        self.training_complete = False
        self.mapping_tau = 20

    def setup(self) -> None:
        """Set up the aligner"""
        super().setup()
        self.previous_aligner.align()
        self.acoustic_model.export_model(self.adapt_directory)
        for f in ["final.mdl", "final.alimdl"]:
            p = os.path.join(self.adapt_directory, f)
            if not os.path.exists(p):
                continue
            os.rename(p, os.path.join(self.adapt_directory, f.replace("final", "0")))

    @property
    def align_directory(self) -> str:
        """Align directory"""
        return os.path.join(self.temp_directory, "adapted_align")

    @property
    def adapt_directory(self) -> str:
        """Adapt directory"""
        return os.path.join(self.temp_directory, "adapt")

    @property
    def working_directory(self) -> str:
        """Current working directory"""
        if self.training_complete:
            return self.align_directory
        return self.adapt_directory

    @property
    def working_log_directory(self) -> str:
        """Current log directory"""
        return os.path.join(self.working_directory, "log")

    @property
    def current_model_path(self):
        """Current acoustic model path"""
        if self.training_complete:
            return os.path.join(self.working_directory, "final.mdl")
        return os.path.join(self.working_directory, "0.mdl")

    @property
    def next_model_path(self):
        """Next iteration's acoustic model path"""
        return os.path.join(self.working_directory, "final.mdl")

    def train(self) -> None:
        """Run the adaptation"""
        done_path = os.path.join(self.adapt_directory, "done")
        dirty_path = os.path.join(self.adapt_directory, "dirty")
        if os.path.exists(done_path):
            self.logger.info("Adapting already done, skipping.")
            return
        try:
            self.logger.info("Adapting pretrained model...")
            train_map(self)
            self.training_complete = True
            shutil.copyfile(
                os.path.join(self.adapt_directory, "final.mdl"),
                os.path.join(self.align_directory, "final.mdl"),
            )
            shutil.copyfile(
                os.path.join(self.adapt_directory, "final.occs"),
                os.path.join(self.align_directory, "final.occs"),
            )
            shutil.copyfile(
                os.path.join(self.adapt_directory, "tree"),
                os.path.join(self.align_directory, "tree"),
            )
            if os.path.exists(os.path.join(self.adapt_directory, "final.alimdl")):
                shutil.copyfile(
                    os.path.join(self.adapt_directory, "final.alimdl"),
                    os.path.join(self.align_directory, "final.alimdl"),
                )
            if os.path.exists(os.path.join(self.adapt_directory, "lda.mat")):
                shutil.copyfile(
                    os.path.join(self.adapt_directory, "lda.mat"),
                    os.path.join(self.align_directory, "lda.mat"),
                )
        except Exception as e:
            with open(dirty_path, "w"):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise
        with open(done_path, "w"):
            pass

    @property
    def meta(self) -> MetaDict:
        """Acoustic model metadata"""
        from datetime import datetime

        from ..utils import get_mfa_version

        data = {
            "phones": sorted(self.dictionary.config.non_silence_phones),
            "version": get_mfa_version(),
            "architecture": self.acoustic_model.meta["architecture"],
            "train_date": str(datetime.now()),
            "features": self.previous_aligner.align_config.feature_config.params(),
            "multilingual_ipa": self.dictionary.config.multilingual_ipa,
        }
        if self.dictionary.config.multilingual_ipa:
            data["strip_diacritics"] = self.dictionary.config.strip_diacritics
            data["digraphs"] = self.dictionary.config.digraphs
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

    def align(self, subset: Optional[int] = None) -> None:
        """
        Align using the adapted model

        Parameters
        ----------
        subset: int, optional
            Number of utterances to align in corpus
        """
        done_path = os.path.join(self.align_directory, "done")
        dirty_path = os.path.join(self.align_directory, "dirty")
        if os.path.exists(done_path):
            self.logger.info("Alignment already done, skipping.")
            return
        try:
            log_dir = os.path.join(self.align_directory, "log")
            os.makedirs(log_dir, exist_ok=True)
            compile_train_graphs(self)

            self.logger.info("Performing first-pass alignment...")
            self.speaker_independent = True
            align(self)
            unaligned, average_log_like = compile_information(self)
            self.logger.debug(
                f"Prior to SAT, average per frame likelihood (this might not actually mean anything): {average_log_like}"
            )
            if (
                not self.align_config.disable_sat
                and self.previous_aligner.acoustic_model.feature_config.fmllr
                and not os.path.exists(os.path.join(self.align_directory, "trans.0"))
            ):
                self.logger.info("Calculating fMLLR for speaker adaptation...")
                calc_fmllr(self)

                self.speaker_independent = False
                self.logger.info("Performing second-pass alignment...")
                align(self)

                unaligned, average_log_like = compile_information(self)
                self.logger.debug(
                    f"Following SAT, average per frame likelihood (this might not actually mean anything): {average_log_like}"
                )

        except Exception as e:
            with open(dirty_path, "w"):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise
        with open(done_path, "w"):
            pass
