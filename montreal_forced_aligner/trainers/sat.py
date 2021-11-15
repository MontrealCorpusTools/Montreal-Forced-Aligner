"""Class definitions for Speaker Adapted Triphone trainer"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import TYPE_CHECKING, Optional

from ..abc import MetaDict
from ..exceptions import KaldiProcessingError
from ..multiprocessing import (
    acc_stats,
    align,
    calc_fmllr,
    compile_information,
    compile_train_graphs,
    compute_alignment_improvement,
    convert_alignments,
    create_align_model,
    tree_stats,
)
from ..utils import log_kaldi_errors, parse_logs, thirdparty_binary
from .triphone import TriphoneTrainer

if TYPE_CHECKING:
    from ..abc import Dictionary, Trainer
    from ..config import FeatureConfig
    from ..corpus import Corpus


__all__ = ["SatTrainer"]


class SatTrainer(TriphoneTrainer):
    """

    Configuration class for speaker adapted training (SAT)

    Attributes
    ----------
    fmllr_update_type : str
        Type of fMLLR estimation, defaults to ``'full'``
    fmllr_iterations : list
        List of iterations to perform fMLLR estimation
    silence_weight : float
        Weight on silence in fMLLR estimation
    """

    def __init__(self, default_feature_config: FeatureConfig):
        super(SatTrainer, self).__init__(default_feature_config)
        self.fmllr_update_type = "full"
        self.fmllr_iterations = []
        max_fmllr_iter = int(self.num_iterations / 2) - 1
        for i in range(1, max_fmllr_iter):
            if i < max_fmllr_iter / 2 and i % 2 == 0:
                self.fmllr_iterations.append(i)
        self.fmllr_iterations.append(max_fmllr_iter)
        self.silence_weight = 0.0
        self.feature_config.fmllr = True
        self.initial_fmllr = True
        self.ensure_train = True

    def compute_calculated_properties(self) -> None:
        """Generate realignment iterations, initial gaussians, and fMLLR iteraction based on configuration"""
        super(SatTrainer, self).compute_calculated_properties()
        self.fmllr_iterations = []
        max_fmllr_iter = int(self.num_iterations / 2) - 1
        for i in range(1, max_fmllr_iter):
            if i < max_fmllr_iter / 2 and i % 2 == 0:
                self.fmllr_iterations.append(i)
        self.fmllr_iterations.append(max_fmllr_iter)

    @property
    def train_type(self) -> str:
        """Training identifier"""
        return "sat"

    @property
    def fmllr_options(self) -> MetaDict:
        """Options for calculating fMLLR transforms"""
        return {
            "fmllr_update_type": self.fmllr_update_type,
            "debug": self.debug,
            "initial": self.initial_fmllr,
            "silence_csl": self.dictionary.config.silence_csl,
        }

    @property
    def working_directory(self) -> str:
        """Current working directory"""
        if self.ensure_train:
            return self.train_directory
        return super().working_directory

    @property
    def working_log_directory(self) -> str:
        """Current log directory"""
        if self.ensure_train:
            return self.log_directory
        return super().working_log_directory

    def finalize_training(self) -> None:
        """
        Finalize training and create a speaker independent model for initial alignment

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        try:
            super().finalize_training()
            create_align_model(self)
            self.ensure_train = False
            shutil.copyfile(
                os.path.join(self.train_directory, "final.alimdl"),
                os.path.join(self.align_directory, "final.alimdl"),
            )
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise

    def training_iteration(self) -> None:
        """
        Run a single training iteration
        """
        if os.path.exists(self.next_model_path):
            self.iteration += 1
            return
        if self.iteration in self.realignment_iterations:
            align(self)
            if self.debug:
                compute_alignment_improvement(self)
        if self.iteration in self.fmllr_iterations:
            calc_fmllr(self)

        acc_stats(self)
        parse_logs(self.log_directory)
        if self.iteration < self.final_gaussian_iteration:
            self.increment_gaussians()
        self.iteration += 1

    def align(self, subset: Optional[int] = None) -> None:
        """
        Align a given subset of the corpus

        Parameters
        ----------
        subset: int, optional
            Number of utterances to select for the aligned subset

        Raises
        ------
        KaldiProcessingError
            If there were any errors in running Kaldi binaries
        """
        if not os.path.exists(self.align_directory):
            self.finalize_training()
        dirty_path = os.path.join(self.align_directory, "dirty")
        if os.path.exists(dirty_path):  # if there was an error, let's redo from scratch
            shutil.rmtree(self.align_directory)
        done_path = os.path.join(self.align_directory, "done")
        if not os.path.exists(done_path):
            message = f"Generating alignments using {self.identifier} models"
            if subset:
                message += f" using {subset} utterances..."
            else:
                message += " for the whole corpus..."
            self.logger.info(message)
            begin = time.time()
            if subset is None:
                self.data_directory = self.corpus.split_directory
            else:
                self.data_directory = self.corpus.subset_directory(subset)
            try:
                self.speaker_independent = True
                self.initial_fmllr = True
                compile_train_graphs(self)
                align(self)

                unaligned, average_log_like = compile_information(self)
                self.logger.debug(
                    f"Before SAT, average per frame likelihood (this might not actually mean anything): {average_log_like}"
                )

                if self.speaker_independent:
                    calc_fmllr(self)
                    self.speaker_independent = False
                    self.initial_fmllr = False
                    align(self)
                self.save(os.path.join(self.align_directory, "acoustic_model.zip"))

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
            self.logger.debug(f"Alignment took {time.time() - begin} seconds")
        else:
            self.logger.info(f"Alignments using {self.identifier} models already done")

    def init_training(
        self,
        identifier: str,
        temporary_directory: str,
        corpus: Corpus,
        dictionary: Dictionary,
        previous_trainer: Optional[Trainer],
    ) -> None:
        """
        Initialize speaker-adapted triphone training

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
        """
        self.feature_config.fmllr = False
        self._setup_for_init(identifier, temporary_directory, corpus, dictionary, previous_trainer)
        done_path = os.path.join(self.train_directory, "done")
        dirty_path = os.path.join(self.train_directory, "dirty")
        self.feature_config.fmllr = True
        if os.path.exists(done_path):
            self.logger.info(f"{self.identifier} training already done, skipping initialization.")
            return
        if os.path.exists(os.path.join(self.train_directory, "1.mdl")):
            return
        begin = time.time()
        self.logger.info("Initializing speaker-adapted triphone training...")
        align_directory = previous_trainer.align_directory
        try:
            if os.path.exists(os.path.join(align_directory, "lda.mat")):
                shutil.copyfile(
                    os.path.join(align_directory, "lda.mat"),
                    os.path.join(self.train_directory, "lda.mat"),
                )
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

            with open(extra_question_int_path, "r") as in_file, open(
                questions_path, "a"
            ) as out_file:
                for line in in_file:
                    out_file.write(line)

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
            mdl_path = os.path.join(self.train_directory, "0.mdl")
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

            compile_train_graphs(self)

            convert_alignments(self)

            if os.path.exists(os.path.join(align_directory, "trans.0.ark")):
                for j in self.corpus.jobs:
                    for path in j.construct_path_dictionary(
                        align_directory, "trans", "ark"
                    ).values():
                        shutil.copy(path, path.replace(align_directory, self.train_directory))
            else:

                calc_fmllr(self)
            self.initial_fmllr = False
            self.iteration = 1
            os.rename(occs_path, os.path.join(self.train_directory, "1.occs"))
            os.rename(mdl_path, os.path.join(self.train_directory, "1.mdl"))
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
