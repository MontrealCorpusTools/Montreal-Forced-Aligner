"""Class definitions for TriphoneTrainer"""
from __future__ import annotations

import logging
import os
import typing
from pathlib import Path
from typing import Dict, List

from _kalpy.gmm import gmm_init_model, gmm_init_model_from_previous
from _kalpy.hmm import convert_alignments
from _kalpy.tree import automatically_obtain_questions, build_tree
from _kalpy.util import Int32VectorWriter
from kalpy.gmm.data import AlignmentArchive
from kalpy.gmm.train import TreeStatsAccumulator
from kalpy.gmm.utils import read_gmm_model, read_topology, read_tree
from kalpy.utils import generate_write_specifier, kalpy_logger
from sqlalchemy.orm import joinedload, subqueryload

from montreal_forced_aligner import config
from montreal_forced_aligner.abc import KaldiFunction, MetaDict
from montreal_forced_aligner.acoustic_modeling.base import AcousticModelTrainingMixin
from montreal_forced_aligner.data import MfaArguments, PhoneType
from montreal_forced_aligner.db import Job, Phone
from montreal_forced_aligner.utils import run_kaldi_function, thread_logger

__all__ = [
    "TriphoneTrainer",
    "TreeStatsArguments",
    "ConvertAlignmentsFunction",
    "ConvertAlignmentsArguments",
]

logger = logging.getLogger("mfa")


class TreeStatsArguments(MfaArguments):
    """Arguments for :func:`~montreal_forced_aligner.acoustic_modeling.triphone.tree_stats_func`"""

    working_directory: Path
    model_path: Path


class ConvertAlignmentsArguments(MfaArguments):
    """Arguments for :func:`~montreal_forced_aligner.acoustic_modeling.triphone.ConvertAlignmentsFunction`"""

    dictionaries: List[str]
    model_path: Path
    tree_path: Path
    align_model_path: Path
    ali_paths: Dict[str, Path]
    new_ali_paths: Dict[str, Path]


class ConvertAlignmentsFunction(KaldiFunction):
    """
    Multiprocessing function for converting alignments from a previous trainer

    See Also
    --------
    :meth:`.TriphoneTrainer.convert_alignments`
        Main function that calls this function in parallel
    :meth:`.TriphoneTrainer.convert_alignments_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`convert-ali`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.acoustic_modeling.triphone.ConvertAlignmentsArguments`
        Arguments for the function
    """

    def __init__(self, args: ConvertAlignmentsArguments):
        super().__init__(args)
        self.dictionaries = args.dictionaries
        self.model_path = args.model_path
        self.tree_path = args.tree_path
        self.align_model_path = args.align_model_path
        self.ali_paths = args.ali_paths
        self.new_ali_paths = args.new_ali_paths

    def _run(self) -> None:
        """Run the function"""
        with self.session() as session, thread_logger(
            "kalpy.train", self.log_path, job_name=self.job_name
        ) as train_logger:
            job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )
            train_logger.debug(f"Previous model path: {self.align_model_path}")
            train_logger.debug(f"Model path: {self.model_path}")
            train_logger.debug(f"Tree path: {self.tree_path}")
            for d in job.training_dictionaries:
                dict_id = d.id
                train_logger.debug(f"Converting alignments for {d.name}")
                ali_path = self.ali_paths[dict_id]
                if not ali_path.exists():
                    continue
                new_ali_path = self.new_ali_paths[dict_id]
                train_logger.debug(f"Old alignments: {ali_path}")
                train_logger.debug(f"New alignments: {new_ali_path}")
                tree = read_tree(self.tree_path)
                old_transition_model, _ = read_gmm_model(self.align_model_path)
                new_transition_model, _ = read_gmm_model(self.model_path)
                alignment_archive = AlignmentArchive(ali_path)
                new_alignment_writer = Int32VectorWriter(generate_write_specifier(new_ali_path))
                for old_alignment in alignment_archive:
                    new_alignment = convert_alignments(
                        old_transition_model,
                        new_transition_model,
                        tree,
                        old_alignment.alignment,
                    )
                    new_alignment_writer.Write(old_alignment.utterance_id, new_alignment)
                    self.callback(old_alignment.utterance_id)
                new_alignment_writer.Close()


class TreeStatsFunction(KaldiFunction):
    """
    Multiprocessing function for calculating tree stats for training

    See Also
    --------
    :meth:`.TriphoneTrainer.tree_stats`
        Main function that calls this function in parallel
    :meth:`.TriphoneTrainer.tree_stats_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`acc-tree-stats`
        Relevant Kaldi binary

    Parameters
    ----------
    args: TreeStatsArguments
        Arguments for the function
    """

    def __init__(self, args: TreeStatsArguments):
        super().__init__(args)
        self.working_directory = args.working_directory
        self.model_path = args.model_path

    def _run(self):
        """Run the function"""
        with self.session() as session, thread_logger(
            "kalpy.train", self.log_path, job_name=self.job_name
        ) as train_logger:
            job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )
            silence_phones = [
                x
                for x, in session.query(Phone.mapping_id)
                .filter(Phone.phone_type.in_([PhoneType.silence, PhoneType.oov]))
                .order_by(Phone.mapping_id)
            ]
            for d in job.training_dictionaries:
                train_logger.debug(f"Accumulating stats for dictionary {d.name} ({d.id})")
                train_logger.debug(f"Accumulating stats for model: {self.model_path}")
                dict_id = d.id
                feature_archive = job.construct_feature_archive(self.working_directory, dict_id)
                ali_path = job.construct_path(self.working_directory, "ali", "ark", dict_id)
                if not ali_path.exists():
                    continue
                train_logger.debug("Feature Archive information:")
                train_logger.debug(f"File: {feature_archive.file_name}")
                train_logger.debug(f"CMVN: {feature_archive.cmvn_read_specifier}")
                train_logger.debug(f"Deltas: {feature_archive.use_deltas}")
                train_logger.debug(f"Splices: {feature_archive.use_splices}")
                train_logger.debug(f"LDA: {feature_archive.lda_mat_file_name}")
                train_logger.debug(f"fMLLR: {feature_archive.transform_read_specifier}")
                train_logger.debug(f"Alignment path: {ali_path}")
                alignment_archive = AlignmentArchive(ali_path)
                accumulator = TreeStatsAccumulator(
                    self.model_path, context_independent_symbols=silence_phones
                )
                accumulator.accumulate_stats(
                    feature_archive, alignment_archive, callback=self.callback
                )
                self.callback(accumulator.tree_stats)


class TriphoneTrainer(AcousticModelTrainingMixin):
    """
    Triphone trainer

    Parameters
    ----------
    subset : int
        Number of utterances to use, defaults to 5000
    num_iterations : int
        Number of training iterations to perform, defaults to 35
    num_leaves : int
        Number of states in the decision tree, defaults to 1000
    max_gaussians : int
        Number of gaussians in the decision tree, defaults to 10000
    cluster_threshold : int
        For build-tree control final bottom-up clustering of leaves, defaults to 100

    See Also
    --------
    :class:`~montreal_forced_aligner.acoustic_modeling.base.AcousticModelTrainingMixin`
        For acoustic model training parsing parameters
    """

    def __init__(
        self,
        subset: int = 5000,
        num_iterations: int = 35,
        num_leaves: int = 1000,
        max_gaussians: int = 10000,
        cluster_threshold: int = -1,
        boost_silence: float = 1.25,
        power: float = 0.25,
        **kwargs,
    ):
        kwargs["initial_gaussians"] = num_leaves
        super().__init__(
            num_iterations=num_iterations,
            boost_silence=boost_silence,
            power=power,
            subset=subset,
            max_gaussians=max_gaussians,
            **kwargs,
        )
        self.num_leaves = num_leaves
        self.cluster_threshold = cluster_threshold

    def tree_stats_arguments(self) -> List[TreeStatsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.acoustic_modeling.triphone.tree_stats_func`


        Returns
        -------
        list[:class:`~montreal_forced_aligner.acoustic_modeling.triphone.TreeStatsArguments`]
            Arguments for processing
        """
        alignment_model_path = os.path.join(self.previous_aligner.working_directory, "final.mdl")
        arguments = []
        for j in self.jobs:
            arguments.append(
                TreeStatsArguments(
                    j.id,
                    getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                    self.working_log_directory.joinpath(f"acc_tree.{j.id}.log"),
                    self.previous_aligner.working_directory,
                    alignment_model_path,
                )
            )
        return arguments

    def convert_alignments_arguments(self) -> List[ConvertAlignmentsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.acoustic_modeling.triphone.ConvertAlignmentsFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.acoustic_modeling.triphone.ConvertAlignmentsArguments`]
            Arguments for processing
        """
        return [
            ConvertAlignmentsArguments(
                j.id,
                getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                self.working_log_directory.joinpath(f"convert_alignments.{j.id}.log"),
                j.dictionary_ids,
                self.model_path,
                self.tree_path,
                self.previous_aligner.model_path,
                j.construct_path_dictionary(self.previous_aligner.working_directory, "ali", "ark"),
                j.construct_path_dictionary(self.working_directory, "ali", "ark"),
            )
            for j in self.jobs
        ]

    def convert_alignments(self) -> None:
        """
        Multiprocessing function that converts alignments from previous training

        See Also
        --------
        :func:`~montreal_forced_aligner.acoustic_modeling.triphone.ConvertAlignmentsFunction`
            Multiprocessing helper function for each job
        :meth:`.TriphoneTrainer.convert_alignments_arguments`
            Job method for generating arguments for the helper function
        :kaldi_steps:`train_deltas`
            Reference Kaldi script
        :kaldi_steps:`train_lda_mllt`
            Reference Kaldi script
        :kaldi_steps:`train_sat`
            Reference Kaldi script

        """
        logger.info("Converting alignments...")
        arguments = self.convert_alignments_arguments()
        for _ in run_kaldi_function(
            ConvertAlignmentsFunction, arguments, total_count=self.num_current_utterances
        ):
            pass

    def acoustic_model_training_params(self) -> MetaDict:
        """Configuration parameters"""
        return {
            "num_iterations": self.num_iterations,
            "num_leaves": self.num_leaves,
            "max_gaussians": self.max_gaussians,
            "cluster_threshold": self.cluster_threshold,
        }

    def compute_calculated_properties(self) -> None:
        """Generate realignment iterations and initial gaussians based on configuration"""
        for i in range(0, self.num_iterations, 10):
            if i == 0:
                continue
            self.realignment_iterations.append(i)
        self.initial_gaussians = self.num_leaves
        self.final_gaussian_iteration = self.num_iterations - 10

    @property
    def train_type(self) -> str:
        """Training identifier"""
        return "tri"

    @property
    def phone_type(self) -> str:
        """Phone type"""
        return "triphone"

    def _trainer_initialization(self) -> None:
        """Triphone training initialization"""
        if self.initialized:
            return
        self._setup_tree()

        self.compile_train_graphs()

        self.convert_alignments()
        os.rename(self.model_path, self.next_model_path)

    def tree_stats(self) -> typing.List:
        """
        Multiprocessing function that computes stats for decision tree training.

        See Also
        --------
        :func:`~montreal_forced_aligner.acoustic_modeling.triphone.tree_stats_func`
            Multiprocessing helper function for each job
        :meth:`.TriphoneTrainer.tree_stats_arguments`
            Job method for generating arguments for the helper function
        :kaldi_src:`sum-tree-stats`
            Relevant Kaldi binary
        :kaldi_steps:`train_deltas`
            Reference Kaldi script
        :kaldi_steps:`train_lda_mllt`
            Reference Kaldi script
        :kaldi_steps:`train_sat`
            Reference Kaldi script

        """
        logger.info("Accumulating tree stats...")
        arguments = self.tree_stats_arguments()
        tree_stats = {}
        for result in run_kaldi_function(
            TreeStatsFunction, arguments, total_count=self.num_current_utterances
        ):
            if isinstance(result, dict):
                for k, v in result.items():
                    if k not in tree_stats:
                        tree_stats[k] = v
                    else:
                        tree_stats[k].Add(v)
        tree_stats = [(list(k), v) for k, v in tree_stats.items()]
        return tree_stats

    def _setup_tree(self, init_from_previous=False, initial_mix_up=True) -> None:
        """
        Set up the tree for the triphone model

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        log_path = self.working_log_directory.joinpath("setup_tree.log")
        tree_stats = self.tree_stats()
        phone_sets = self.worker.shared_phones_set_symbols()
        roots_int_path = os.path.join(self.worker.phones_dir, "roots.int")
        topo_path = self.worker.topo_path
        topo = read_topology(topo_path)
        with kalpy_logger("kalpy.train", log_path) as train_logger:
            train_logger.debug(f"Topo path: {topo_path}")
            train_logger.debug(f"Tree path: {self.tree_path}")
            train_logger.debug(f"Phone sets: {phone_sets}")
            questions = automatically_obtain_questions(tree_stats, phone_sets, [1], 1)
            train_logger.debug(f"Automatically obtained {len(questions)} questions")
            train_logger.debug("Automatic questions:")
            for q_set in questions:
                train_logger.debug(", ".join([self.reversed_phone_mapping[x] for x in q_set]))

            extra_questions = self.worker.extra_questions_mapping
            if extra_questions:
                train_logger.debug(f"Adding {len(extra_questions)} questions")
                train_logger.debug("Extra questions:")
                for v in self.worker.extra_questions_mapping.values():
                    questions.append(sorted([self.phone_mapping[x] for x in v]))
                    train_logger.debug(", ".join(v))
            train_logger.debug(f"{len(questions)} total questions")

            build_tree(
                topo,
                questions,
                tree_stats,
                str(roots_int_path),
                str(self.tree_path),
                max_leaves=self.num_leaves,
                cluster_thresh=self.cluster_threshold,
            )
            tree = read_tree(self.tree_path)
            mix_up = 0
            mix_down = 0
            if init_from_previous:
                if initial_mix_up:
                    mix_up = self.initial_gaussians
                    mix_down = self.initial_gaussians
                train_logger.debug(f"Mixing up: {mix_up}")
                train_logger.debug(f"Mixing down: {mix_down}")
                old_transition_model, old_acoustic_model = read_gmm_model(
                    os.path.join(self.previous_aligner.working_directory, "final.mdl")
                )
                old_tree = read_tree(os.path.join(self.previous_aligner.working_directory, "tree"))
                gmm_init_model_from_previous(
                    topo,
                    tree,
                    tree_stats,
                    old_acoustic_model,
                    old_transition_model,
                    old_tree,
                    str(self.model_path),
                    mixup=mix_up,
                    mixdown=mix_down,
                )
            else:
                if initial_mix_up:
                    mix_up = self.initial_gaussians
                train_logger.debug(f"Mixing up: {mix_up}")
                train_logger.debug(f"Mixing down: {mix_down}")
                gmm_init_model(
                    topo, tree, tree_stats, str(self.model_path), mixup=mix_up, mixdown=mix_down
                )
