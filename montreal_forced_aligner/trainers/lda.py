"""Class definitions for LDA trainer"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Optional

from ..abc import MetaDict, Trainer
from ..exceptions import KaldiProcessingError
from ..multiprocessing import (
    acc_stats,
    align,
    calc_lda_mllt,
    compute_alignment_improvement,
    lda_acc_stats,
)
from ..utils import log_kaldi_errors, parse_logs
from .triphone import TriphoneTrainer

if TYPE_CHECKING:
    from ..config import FeatureConfig
    from ..corpus import Corpus
    from ..dictionary import MultispeakerDictionary


__all__ = ["LdaTrainer"]


class LdaTrainer(TriphoneTrainer):
    """

    Configuration class for LDA+MLLT training

    Attributes
    ----------
    lda_dimension : int
        Dimensionality of the LDA matrix
    mllt_iterations : list
        List of iterations to perform MLLT estimation
    random_prune : float
        This is approximately the ratio by which we will speed up the
        LDA and MLLT calculations via randomized pruning
    """

    def __init__(self, default_feature_config: FeatureConfig):
        super(LdaTrainer, self).__init__(default_feature_config)
        self.lda_dimension = 40
        self.mllt_iterations = []
        max_mllt_iter = int(self.num_iterations / 2) - 1
        for i in range(1, max_mllt_iter):
            if i < max_mllt_iter / 2 and i % 2 == 0:
                self.mllt_iterations.append(i)
        self.mllt_iterations.append(max_mllt_iter)
        if not self.mllt_iterations:
            self.mllt_iterations = range(1, 4)
        self.random_prune = 4.0

        self.feature_config.lda = True
        self.feature_config.deltas = True
        self.uses_splices = True

    def compute_calculated_properties(self) -> None:
        """Generate realignment iterations, MLLT estimation iterations, and initial gaussians based on configuration"""
        super(LdaTrainer, self).compute_calculated_properties()
        self.mllt_iterations = []
        max_mllt_iter = int(self.num_iterations / 2) - 1
        for i in range(1, max_mllt_iter):
            if i < max_mllt_iter / 2 and i % 2 == 0:
                self.mllt_iterations.append(i)
        self.mllt_iterations.append(max_mllt_iter)

    @property
    def train_type(self) -> str:
        """Training identifier"""
        return "lda"

    @property
    def lda_options(self) -> MetaDict:
        """Options for computing LDA"""
        return {
            "lda_dimension": self.lda_dimension,
            "boost_silence": self.boost_silence,
            "random_prune": self.random_prune,
            "silence_csl": self.dictionary.config.silence_csl,
        }

    def init_training(
        self,
        identifier: str,
        temporary_directory: str,
        corpus: Corpus,
        dictionary: MultispeakerDictionary,
        previous_trainer: Optional[Trainer],
    ):
        """
        Initialize LDA training

        Parameters
        ----------
        identifier: str
            Identifier for the training block
        temporary_directory: str
            Root temporary directory to save
        corpus: :class:`~montreal_forced_aligner.corpus.Corpus`
            Corpus to use
        dictionary: Dictionary
            Pronunciation dictionary to use
        previous_trainer: Trainer, optional
            Previous trainer to initialize from

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        self._setup_for_init(identifier, temporary_directory, corpus, dictionary, previous_trainer)
        done_path = os.path.join(self.train_directory, "done")
        dirty_path = os.path.join(self.train_directory, "dirty")
        if os.path.exists(done_path):
            self.logger.info("{self.identifier} training already done, skipping initialization.")
            return
        begin = time.time()
        try:
            self.feature_config.directory = None
            lda_acc_stats(self)
            self.feature_config.directory = self.train_directory
        except Exception as e:
            with open(dirty_path, "w") as _:
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise
        self._setup_tree()
        self.iteration = 1
        self.logger.info("Initialization complete!")
        self.logger.debug(f"Initialization took {time.time() - begin} seconds")

    def training_iteration(self):
        """
        Run a single training iteration
        """
        if os.path.exists(self.next_model_path):
            return
        if self.iteration in self.realignment_iterations:
            align(self)
            if self.debug:
                compute_alignment_improvement(self)
        if self.iteration in self.mllt_iterations:
            calc_lda_mllt(self)

        acc_stats(self)
        parse_logs(self.log_directory)
        if self.iteration < self.final_gaussian_iteration:
            self.increment_gaussians()
        self.iteration += 1
