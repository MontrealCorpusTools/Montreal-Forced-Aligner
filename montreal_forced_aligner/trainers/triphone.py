"""Class definitions for TriphoneTrainer"""
from __future__ import annotations

import os
import subprocess
import time
from typing import TYPE_CHECKING, Optional

from ..exceptions import KaldiProcessingError
from ..multiprocessing import compile_train_graphs, convert_alignments, tree_stats
from ..utils import log_kaldi_errors, parse_logs, thirdparty_binary
from .base import BaseTrainer

if TYPE_CHECKING:
    from ..abc import Dictionary, Trainer
    from ..config import FeatureConfig
    from ..corpus import Corpus


__all__ = ["TriphoneTrainer"]


class TriphoneTrainer(BaseTrainer):
    """
    Configuration class for triphone training

    Attributes
    ----------
    num_iterations : int
        Number of training iterations to perform, defaults to 40
    num_leaves : int
        Number of states in the decision tree, defaults to 1000
    max_gaussians : int
        Number of gaussians in the decision tree, defaults to 10000
    cluster_threshold : int
        For build-tree control final bottom-up clustering of leaves, defaults to 100
    """

    def __init__(self, default_feature_config: FeatureConfig):
        super(TriphoneTrainer, self).__init__(default_feature_config)

        self.num_iterations = 35
        self.num_leaves = 1000
        self.max_gaussians = 10000
        self.cluster_threshold = -1
        self.compute_calculated_properties()

    def compute_calculated_properties(self) -> None:
        """Generate realignment iterations and initial gaussians based on configuration"""
        for i in range(0, self.num_iterations, 10):
            if i == 0:
                continue
            self.realignment_iterations.append(i)
        self.initial_gaussians = self.num_leaves
        self.current_gaussians = self.num_leaves

    @property
    def train_type(self) -> str:
        """Training identifier"""
        return "tri"

    @property
    def phone_type(self) -> str:
        """Phone type"""
        return "triphone"

    def _setup_tree(self) -> None:
        """
        Set up the tree for the triphone model

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        dirty_path = os.path.join(self.train_directory, "dirty")
        try:

            tree_stats(self)
            log_path = os.path.join(self.log_directory, "questions.log")
            tree_path = os.path.join(self.train_directory, "tree")
            treeacc_path = os.path.join(self.train_directory, "treeacc")
            sets_int_path = os.path.join(self.dictionary.phones_dir, "sets.int")
            roots_int_path = os.path.join(self.dictionary.phones_dir, "roots.int")
            extra_question_int_path = os.path.join(
                self.dictionary.phones_dir, "extra_questions.int"
            )
            topo_path = self.dictionary.topo_path
            questions_path = os.path.join(self.train_directory, "questions.int")
            questions_qst_path = os.path.join(self.train_directory, "questions.qst")
            with open(log_path, "w") as log_file:
                subprocess.call(
                    [
                        thirdparty_binary("cluster-phones"),
                        treeacc_path,
                        sets_int_path,
                        questions_path,
                    ],
                    stderr=log_file,
                )

            with open(extra_question_int_path, "r") as inf, open(questions_path, "a") as outf:
                for line in inf:
                    outf.write(line)

            log_path = os.path.join(self.log_directory, "compile_questions.log")
            with open(log_path, "w") as log_file:
                subprocess.call(
                    [
                        thirdparty_binary("compile-questions"),
                        topo_path,
                        questions_path,
                        questions_qst_path,
                    ],
                    stderr=log_file,
                )

            log_path = os.path.join(self.log_directory, "build_tree.log")
            with open(log_path, "w") as log_file:
                subprocess.call(
                    [
                        thirdparty_binary("build-tree"),
                        "--verbose=1",
                        f"--max-leaves={self.initial_gaussians}",
                        f"--cluster-thresh={self.cluster_threshold}",
                        treeacc_path,
                        roots_int_path,
                        questions_qst_path,
                        topo_path,
                        tree_path,
                    ],
                    stderr=log_file,
                )

            log_path = os.path.join(self.log_directory, "init_model.log")
            occs_path = os.path.join(self.train_directory, "0.occs")
            mdl_path = self.current_model_path
            with open(log_path, "w") as log_file:
                subprocess.call(
                    [
                        thirdparty_binary("gmm-init-model"),
                        f"--write-occs={occs_path}",
                        tree_path,
                        treeacc_path,
                        topo_path,
                        mdl_path,
                    ],
                    stderr=log_file,
                )

            log_path = os.path.join(self.log_directory, "mixup.log")
            with open(log_path, "w") as log_file:
                subprocess.call(
                    [
                        thirdparty_binary("gmm-mixup"),
                        f"--mix-up={self.initial_gaussians}",
                        mdl_path,
                        occs_path,
                        mdl_path,
                    ],
                    stderr=log_file,
                )
            os.remove(treeacc_path)
            parse_logs(self.log_directory)

            compile_train_graphs(self)

            convert_alignments(self)
            os.rename(occs_path, self.next_occs_path)
            os.rename(mdl_path, self.next_model_path)
        except Exception as e:
            with open(dirty_path, "w"):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise

    def init_training(
        self,
        identifier: str,
        temporary_directory: str,
        corpus: Corpus,
        dictionary: Dictionary,
        previous_trainer: Optional[Trainer],
    ):
        """
        Initialize triphone training

        Parameters
        ----------
        identifier: str
            Identifier for the training block
        temporary_directory: str
            Root temporary directory to save
        corpus: :class:`~montreal_forced_aligner.corpus.Corpus`
            Corpus to use
        dictionary: MultispeakerDictionary
            Dictionary to use
        previous_trainer: Trainer, optional
            Previous trainer to initialize from
        """
        self._setup_for_init(identifier, temporary_directory, corpus, dictionary, previous_trainer)
        done_path = os.path.join(self.train_directory, "done")
        if os.path.exists(done_path):
            self.logger.info(f"{self.identifier} training already done, skipping initialization.")
            return
        begin = time.time()
        self._setup_tree()

        self.iteration = 1
        self.logger.info("Initialization complete!")
        self.logger.debug(f"Initialization took {time.time() - begin} seconds")
