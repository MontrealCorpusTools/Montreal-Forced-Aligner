"""Class definitions for Monophone trainer"""
from __future__ import annotations

import os
import re
import subprocess
import time
from typing import TYPE_CHECKING, Optional

from ..exceptions import KaldiProcessingError
from ..multiprocessing import compile_train_graphs, mono_align_equal
from ..utils import log_kaldi_errors, parse_logs, thirdparty_binary
from .base import BaseTrainer

if TYPE_CHECKING:
    from ..config import FeatureConfig
    from ..corpus import Corpus
    from ..dictionary import MultispeakerDictionary


__all__ = ["MonophoneTrainer"]


class MonophoneTrainer(BaseTrainer):
    """
    Configuration class for monophone training


    Attributes
    ----------
    initial_gaussians : int
        Number of gaussians to begin training
    """

    def __init__(self, default_feature_config: FeatureConfig):
        super(MonophoneTrainer, self).__init__(default_feature_config)
        self.initial_gaussians = 135
        self.compute_calculated_properties()

    def compute_calculated_properties(self) -> None:
        """Generate realignment iterations and initial gaussians based on configuration"""
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
    def train_type(self) -> str:
        """Training identifier"""
        return "mono"

    @property
    def phone_type(self) -> str:
        """Phone type"""
        return "monophone"

    def init_training(
        self,
        identifier: str,
        temporary_directory: str,
        corpus: Corpus,
        dictionary: MultispeakerDictionary,
        previous_trainer: Optional[BaseTrainer] = None,
    ) -> None:
        """
        Initialize monophone training

        Parameters
        ----------
        identifier: str
            Identifier for the training block
        temporary_directory: str
            Root temporary directory to save
        corpus: :class:`~montreal_forced_aligner.corpus.Corpus`
            Corpus to use
        dictionary: :class:`~montreal_forced_aligner.dictionary.MultispeakerDictionary`
            MultispeakerDictionary to use
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
            self.logger.info(f"{self.identifier} training already done, skipping initialization.")
            return
        begin = time.time()
        self.iteration = 0
        tree_path = os.path.join(self.train_directory, "tree")

        try:
            feat_dim = corpus.get_feat_dim()

            feature_string = corpus.jobs[0].construct_base_feature_string(corpus)
            shared_phones_path = os.path.join(
                dictionary.get_dictionary("default").phones_dir, "sets.int"
            )
            init_log_path = os.path.join(self.log_directory, "init.log")
            temp_feats_path = os.path.join(self.train_directory, "temp_feats")
            with open(init_log_path, "w") as log_file:
                subprocess.call(
                    [
                        thirdparty_binary("subset-feats"),
                        "--n=10",
                        feature_string,
                        f"ark:{temp_feats_path}",
                    ],
                    stderr=log_file,
                )
                subprocess.call(
                    [
                        thirdparty_binary("gmm-init-mono"),
                        f"--shared-phones={shared_phones_path}",
                        f"--train-feats=ark:{temp_feats_path}",
                        os.path.join(
                            dictionary.get_dictionary("default").output_directory, "topo"
                        ),
                        str(feat_dim),
                        self.current_model_path,
                        tree_path,
                    ],
                    stderr=log_file,
                )
                proc = subprocess.Popen(
                    [thirdparty_binary("gmm-info"), "--print-args=false", self.current_model_path],
                    stderr=log_file,
                    stdout=subprocess.PIPE,
                )
                stdout, stderr = proc.communicate()
                num = stdout.decode("utf8")
                matches = re.search(r"gaussians (\d+)", num)
                num_gauss = int(matches.groups()[0])
            if os.path.exists(self.current_model_path):
                os.remove(init_log_path)
            os.remove(temp_feats_path)
            self.initial_gaussians = num_gauss
            self.current_gaussians = num_gauss
            compile_train_graphs(self)
            mono_align_equal(self)
            self.iteration = 1
            parse_logs(self.log_directory)
        except Exception as e:
            with open(dirty_path, "w"):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise
        self.logger.info("Initialization complete!")
        self.logger.debug(f"Initialization took {time.time() - begin} seconds")
