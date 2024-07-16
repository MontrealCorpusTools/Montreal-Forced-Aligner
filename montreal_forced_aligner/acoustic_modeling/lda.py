"""Class definitions for LDA trainer"""
from __future__ import annotations

import logging
import os
import shutil
import typing
from pathlib import Path

from _kalpy.matrix import FloatMatrix
from _kalpy.transform import LdaEstimateOptions, compose_transforms
from kalpy.feat.data import FeatureArchive
from kalpy.feat.lda import LdaStatsAccumulator, MlltStatsAccumulator
from kalpy.gmm.data import AlignmentArchive
from kalpy.gmm.utils import read_gmm_model, write_gmm_model
from kalpy.utils import kalpy_logger, read_kaldi_object, write_kaldi_object
from sqlalchemy.orm import joinedload

from montreal_forced_aligner import config
from montreal_forced_aligner.abc import KaldiFunction, MetaDict
from montreal_forced_aligner.acoustic_modeling.triphone import TriphoneTrainer
from montreal_forced_aligner.data import MfaArguments, PhoneType
from montreal_forced_aligner.db import Job, Phone
from montreal_forced_aligner.exceptions import TrainerError
from montreal_forced_aligner.utils import parse_logs, run_kaldi_function, thread_logger

__all__ = [
    "LdaTrainer",
    "CalcLdaMlltFunction",
    "CalcLdaMlltArguments",
    "LdaAccStatsFunction",
    "LdaAccStatsArguments",
]

logger = logging.getLogger("mfa")


class LdaAccStatsArguments(MfaArguments):
    """Arguments for :func:`~montreal_forced_aligner.acoustic_modeling.lda.LdaAccStatsFunction`"""

    working_directory: Path
    model_path: Path
    lda_options: MetaDict


class CalcLdaMlltArguments(MfaArguments):
    """Arguments for :func:`~montreal_forced_aligner.acoustic_modeling.lda.CalcLdaMlltFunction`"""

    working_directory: Path
    model_path: Path
    lda_options: MetaDict


class LdaAccStatsFunction(KaldiFunction):
    """
    Multiprocessing function to accumulate LDA stats

    See Also
    --------
    :meth:`.LdaTrainer.lda_acc_stats`
        Main function that calls this function in parallel
    :meth:`.LdaTrainer.lda_acc_stats_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`ali-to-post`
        Relevant Kaldi binary
    :kaldi_src:`weight-silence-post`
        Relevant Kaldi binary
    :kaldi_src:`acc-lda`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.acoustic_modeling.lda.LdaAccStatsArguments`
        Arguments for the function
    """

    def __init__(self, args: LdaAccStatsArguments):
        super().__init__(args)
        self.working_directory = args.working_directory
        self.model_path = args.model_path
        self.lda_options = args.lda_options

    def _run(self):
        """Run the function"""
        with self.session() as session, thread_logger(
            "kalpy.lda", self.log_path, job_name=self.job_name
        ) as lda_logger:
            lda_logger.debug(f"Using acoustic model: {self.model_path}\n")
            job: typing.Optional[Job] = session.get(
                Job, self.job_name, options=[joinedload(Job.dictionaries), joinedload(Job.corpus)]
            )
            silence_phones = [
                x
                for x, in session.query(Phone.mapping_id).filter(
                    Phone.phone_type.in_([PhoneType.silence, PhoneType.oov])
                )
            ]
            for dict_id in job.dictionary_ids:
                ali_path = job.construct_path(self.working_directory, "ali", "ark", dict_id)
                if not ali_path.exists():
                    continue
                lda_logger.debug(f"Processing {ali_path}")
                feat_path = job.construct_path(
                    job.corpus.current_subset_directory, "feats", "scp", dictionary_id=dict_id
                )
                feature_archive = FeatureArchive(
                    feat_path,
                    deltas=False,
                    splices=True,
                    splice_frames=self.lda_options["splice_left_context"],
                )
                alignment_archive = AlignmentArchive(ali_path)
                accumulator = LdaStatsAccumulator(
                    self.model_path, silence_phones, rand_prune=self.lda_options["random_prune"]
                )
                accumulator.accumulate_stats(
                    feature_archive, alignment_archive, callback=self.callback
                )
                self.callback(accumulator.lda)


class CalcLdaMlltFunction(KaldiFunction):
    """
    Multiprocessing function for estimating LDA with MLLT.

    See Also
    --------
    :meth:`.LdaTrainer.calc_lda_mllt`
        Main function that calls this function in parallel
    :meth:`.LdaTrainer.calc_lda_mllt_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`ali-to-post`
        Relevant Kaldi binary
    :kaldi_src:`weight-silence-post`
        Relevant Kaldi binary
    :kaldi_src:`gmm-acc-mllt`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.acoustic_modeling.lda.CalcLdaMlltArguments`
        Arguments for the function
    """

    def __init__(self, args: CalcLdaMlltArguments):
        super().__init__(args)
        self.working_directory = args.working_directory
        self.model_path = args.model_path
        self.lda_options = args.lda_options

    def _run(self) -> None:
        """Run the function"""
        # Estimating MLLT
        with self.session() as session, thread_logger(
            "kalpy.lda", self.log_path, job_name=self.job_name
        ) as lda_logger:
            lda_logger.debug(f"Using acoustic model: {self.model_path}\n")
            job: typing.Optional[Job] = session.get(
                Job, self.job_name, options=[joinedload(Job.dictionaries), joinedload(Job.corpus)]
            )
            silence_phones = [
                x
                for x, in session.query(Phone.mapping_id).filter(
                    Phone.phone_type.in_([PhoneType.silence, PhoneType.oov])
                )
            ]
            for dict_id in job.dictionary_ids:
                ali_path = job.construct_path(self.working_directory, "ali", "ark", dict_id)
                if not ali_path.exists():
                    continue
                lda_logger.debug(f"Processing {ali_path}")
                feature_archive = job.construct_feature_archive(self.working_directory, dict_id)
                alignment_archive = AlignmentArchive(ali_path)
                accumulator = MlltStatsAccumulator(
                    self.model_path, silence_phones, rand_prune=self.lda_options["random_prune"]
                )
                accumulator.accumulate_stats(
                    feature_archive, alignment_archive, callback=self.callback
                )
                self.callback(accumulator.mllt_accs)


class LdaTrainer(TriphoneTrainer):
    """
    Triphone trainer

    Parameters
    ----------
    subset : int
        Number of utterances to use, defaults to 10000
    num_leaves : int
        Number of states in the decision tree, defaults to 2500
    max_gaussians : int
        Number of gaussians in the decision tree, defaults to 15000
    lda_dimension : int
        Dimensionality of the LDA matrix
    uses_splices : bool
        Flag to use spliced and LDA calculation
    splice_left_context : int or None
        Number of frames to splice on the left for calculating LDA
    splice_right_context : int or None
        Number of frames to splice on the right for calculating LDA
    random_prune : float
        This is approximately the ratio by which we will speed up the
        LDA and MLLT calculations via randomized pruning

    See Also
    --------
    :class:`~montreal_forced_aligner.acoustic_modeling.triphone.TriphoneTrainer`
        For acoustic model training parsing parameters

    Attributes
    ----------
    mllt_iterations : list
        List of iterations to perform MLLT estimation
    """

    def __init__(
        self,
        subset: int = 10000,
        num_leaves: int = 2500,
        max_gaussians=15000,
        lda_dimension: int = 40,
        uses_splices: bool = True,
        splice_left_context: int = 3,
        splice_right_context: int = 3,
        random_prune: float = 4.0,
        boost_silence: float = 1.0,
        power: float = 0.25,
        **kwargs,
    ):
        super().__init__(
            boost_silence=boost_silence,
            power=power,
            subset=subset,
            num_leaves=num_leaves,
            max_gaussians=max_gaussians,
            **kwargs,
        )
        self.lda_dimension = lda_dimension
        self.random_prune = random_prune
        self.uses_splices = uses_splices
        self.splice_left_context = splice_left_context
        self.splice_right_context = splice_right_context

    def lda_acc_stats_arguments(self) -> typing.List[LdaAccStatsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.acoustic_modeling.lda.LdaAccStatsFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.acoustic_modeling.lda.LdaAccStatsArguments`]
            Arguments for processing
        """
        arguments = []
        for j in self.jobs:
            arguments.append(
                LdaAccStatsArguments(
                    j.id,
                    getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                    self.working_log_directory.joinpath(f"lda_acc_stats.{j.id}.log"),
                    self.previous_aligner.working_directory,
                    self.previous_aligner.alignment_model_path,
                    self.lda_options,
                )
            )
        return arguments

    def calc_lda_mllt_arguments(self) -> typing.List[CalcLdaMlltArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.acoustic_modeling.lda.CalcLdaMlltFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.acoustic_modeling.lda.CalcLdaMlltArguments`]
            Arguments for processing
        """
        arguments = []
        for j in self.jobs:
            arguments.append(
                CalcLdaMlltArguments(
                    j.id,
                    getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                    os.path.join(
                        self.working_log_directory, f"lda_mllt.{self.iteration}.{j.id}.log"
                    ),
                    self.working_directory,
                    self.model_path,
                    self.lda_options,
                )
            )
        return arguments

    @property
    def train_type(self) -> str:
        """Training identifier"""
        return "lda"

    @property
    def lda_options(self) -> MetaDict:
        """Options for computing LDA"""
        return {
            "lda_dimension": self.lda_dimension,
            "random_prune": self.random_prune,
            "splice_left_context": self.splice_left_context,
            "splice_right_context": self.splice_right_context,
        }

    def compute_calculated_properties(self) -> None:
        """Generate realignment iterations, MLLT estimation iterations, and initial gaussians based on configuration"""
        super().compute_calculated_properties()
        self.mllt_iterations = [2, 4, 6, 12]

    def lda_acc_stats(self) -> None:
        """
        Multiprocessing function that accumulates LDA statistics.

        See Also
        --------
        :func:`~montreal_forced_aligner.acoustic_modeling.lda.LdaAccStatsFunction`
            Multiprocessing helper function for each job
        :meth:`.LdaTrainer.lda_acc_stats_arguments`
            Job method for generating arguments for the helper function
        :kaldi_src:`est-lda`
            Relevant Kaldi binary
        :kaldi_steps:`train_lda_mllt`
            Reference Kaldi script

        """
        logger.info("Calculating initial LDA stats...")
        worker_lda_path = os.path.join(self.worker.working_directory, "lda.mat")
        lda_path = self.working_directory.joinpath("lda.mat")
        if os.path.exists(worker_lda_path):
            os.remove(worker_lda_path)
        arguments = self.lda_acc_stats_arguments()
        lda = None
        for result in run_kaldi_function(
            LdaAccStatsFunction, arguments, total_count=self.num_current_utterances
        ):
            if not isinstance(result, str):
                if lda is None:
                    lda = result
                else:
                    lda.Add(result)

        log_path = self.working_log_directory.joinpath("lda_est.log")

        with kalpy_logger("kalpy.lda", log_path):
            options = LdaEstimateOptions()
            options.dim = self.lda_dimension
            lda_mat, lda_full_mat = lda.estimate(options)
            write_kaldi_object(lda_mat, lda_path)
        shutil.copyfile(
            lda_path,
            worker_lda_path,
        )

    def _trainer_initialization(self) -> None:
        """Initialize LDA training"""
        self.uses_splices = True
        self.worker.uses_splices = True
        if self.initialized:
            return
        self.lda_acc_stats()
        self._setup_tree(initial_mix_up=False)

        self.compile_train_graphs()

        self.convert_alignments()
        os.rename(self.model_path, self.next_model_path)

    def calc_lda_mllt(self) -> None:
        """
        Multiprocessing function that calculates LDA+MLLT transformations.

        See Also
        --------
        :func:`~montreal_forced_aligner.acoustic_modeling.lda.CalcLdaMlltFunction`
            Multiprocessing helper function for each job
        :meth:`.LdaTrainer.calc_lda_mllt_arguments`
            Job method for generating arguments for the helper function
        :kaldi_src:`est-mllt`
            Relevant Kaldi binary
        :kaldi_src:`gmm-transform-means`
            Relevant Kaldi binary
        :kaldi_src:`compose-transforms`
            Relevant Kaldi binary
        :kaldi_steps:`train_lda_mllt`
            Reference Kaldi script

        """
        logger.info("Re-calculating LDA...")
        arguments = self.calc_lda_mllt_arguments()
        mllt_accs = None
        for result in run_kaldi_function(
            CalcLdaMlltFunction, arguments, total_count=self.num_current_utterances
        ):
            if not isinstance(result, str):
                if mllt_accs is None:
                    mllt_accs = result
                else:
                    mllt_accs.Add(result)
        if mllt_accs is None:
            raise TrainerError("No MLLT stats were found")
        log_path = os.path.join(
            self.working_log_directory, f"transform_means.{self.iteration}.log"
        )

        with kalpy_logger("kalpy.lda", log_path) as lda_logger:
            mat, objf_impr, count = mllt_accs.update()
            transition_model, acoustic_model = read_gmm_model(self.model_path)
            lda_logger.debug(
                f"LDA matrix has {mat.NumRows()} rows and {mat.NumCols()} columns "
                f"(acoustic model dimension: {acoustic_model.Dim()})"
            )
            lda_logger.debug(
                f"Overall objective function improvement for MLLT is {objf_impr/count} "
                f"over {count} frames, logdet is {mat.LogDet()}"
            )
            if mat.NumRows() != acoustic_model.Dim():
                raise TrainerError(
                    f"Transform matrix has {mat.NumRows()} rows but "
                    f"model has dimension  {acoustic_model.Dim()}"
                )
            if mat.NumCols() != acoustic_model.Dim() and mat.NumCols() != acoustic_model.Dim() + 1:
                raise TrainerError(
                    f"Transform matrix has {mat.NumCols()} columns but "
                    f"model has dimension {acoustic_model.Dim()} (neither a linear nor an "
                    "affine transform)"
                )
            acoustic_model.transform_means(mat)
            write_gmm_model(self.model_path, transition_model, acoustic_model)
            previous_mat_path = self.working_directory.joinpath("lda.mat")

            prev_mat = read_kaldi_object(FloatMatrix, previous_mat_path)
            new_mat = compose_transforms(mat, prev_mat, False)
            write_kaldi_object(new_mat, str(previous_mat_path))

    def train_iteration(self) -> None:
        """
        Run a single LDA training iteration
        """
        if os.path.exists(self.next_model_path):
            if self.iteration <= self.final_gaussian_iteration:
                self.increment_gaussians()
            self.iteration += 1
            return
        if self.iteration in self.realignment_iterations:
            self.align_iteration()
        if self.iteration in self.mllt_iterations:
            self.calc_lda_mllt()

        self.acc_stats()
        parse_logs(self.working_log_directory)
        if self.iteration <= self.final_gaussian_iteration:
            self.increment_gaussians()
        self.iteration += 1
