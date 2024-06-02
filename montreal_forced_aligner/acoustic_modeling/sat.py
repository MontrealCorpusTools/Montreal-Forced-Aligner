"""Class definitions for Speaker Adapted Triphone trainer"""
from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import List

from _kalpy.gmm import AccumAmDiagGmm
from _kalpy.matrix import DoubleVector
from kalpy.feat.data import FeatureArchive
from kalpy.gmm.data import AlignmentArchive
from kalpy.gmm.train import TwoFeatsStatsAccumulator
from kalpy.gmm.utils import read_gmm_model, write_gmm_model
from kalpy.utils import kalpy_logger
from sqlalchemy.orm import joinedload, subqueryload

from montreal_forced_aligner import config
from montreal_forced_aligner.abc import KaldiFunction
from montreal_forced_aligner.acoustic_modeling.triphone import TriphoneTrainer
from montreal_forced_aligner.data import MfaArguments
from montreal_forced_aligner.db import Job
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.utils import (
    log_kaldi_errors,
    parse_logs,
    run_kaldi_function,
    thread_logger,
)

__all__ = ["SatTrainer", "AccStatsTwoFeatsFunction", "AccStatsTwoFeatsArguments"]


logger = logging.getLogger("mfa")


class AccStatsTwoFeatsArguments(MfaArguments):
    """Arguments for :func:`~montreal_forced_aligner.acoustic_modeling.sat.AccStatsTwoFeatsFunction`"""

    working_directory: Path
    model_path: Path


class AccStatsTwoFeatsFunction(KaldiFunction):
    """
    Multiprocessing function for accumulating stats across speaker-independent and
    speaker-adapted features

    See Also
    --------
    :meth:`.SatTrainer.create_align_model`
        Main function that calls this function in parallel
    :meth:`.SatTrainer.acc_stats_two_feats_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`ali-to-post`
        Relevant Kaldi binary
    :kaldi_src:`gmm-acc-stats-twofeats`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.acoustic_modeling.sat.AccStatsTwoFeatsArguments`
        Arguments for the function
    """

    def __init__(self, args: AccStatsTwoFeatsArguments):
        super().__init__(args)
        self.working_directory = args.working_directory
        self.model_path = args.model_path

    def _run(self):
        """Run the function"""
        with self.session() as session, thread_logger(
            "kalpy.train", self.log_path, job_name=self.job_name
        ) as train_logger:
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )
            for d in job.training_dictionaries:
                train_logger.debug(f"Accumulating stats for dictionary {d.name} ({d.id})")
                train_logger.debug(f"Accumulating stats for model: {self.model_path}")
                dict_id = d.id
                accumulator = TwoFeatsStatsAccumulator(self.model_path)

                ali_path = job.construct_path(self.working_directory, "ali", "ark", dict_id)
                if not ali_path.exists():
                    continue
                fmllr_path = job.construct_path(
                    job.corpus.current_subset_directory, "trans", "scp", dict_id
                )
                if not fmllr_path.exists():
                    fmllr_path = None
                lda_mat_path = self.working_directory.joinpath("lda.mat")
                if not lda_mat_path.exists():
                    lda_mat_path = None
                feat_path = job.construct_path(
                    job.corpus.current_subset_directory, "feats", "scp", dictionary_id=dict_id
                )
                train_logger.debug(f"Feature path: {feat_path}")
                train_logger.debug(f"LDA transform path: {lda_mat_path}")
                train_logger.debug(f"Speaker transform path: {fmllr_path}")
                feature_archive = job.construct_feature_archive(self.working_directory, dict_id)
                si_feature_archive = FeatureArchive(
                    feat_path,
                    lda_mat_file_name=lda_mat_path,
                    deltas=True,
                )
                train_logger.debug("SAT Feature Archive information:")
                train_logger.debug(f"CMVN: {feature_archive.cmvn_read_specifier}")
                train_logger.debug(f"Deltas: {feature_archive.use_deltas}")
                train_logger.debug(f"Splices: {feature_archive.use_splices}")
                train_logger.debug(f"LDA: {feature_archive.lda_mat_file_name}")
                train_logger.debug(f"fMLLR: {feature_archive.transform_read_specifier}")
                train_logger.debug("SI Feature Archive information:")
                train_logger.debug(f"CMVN: {si_feature_archive.cmvn_read_specifier}")
                train_logger.debug(f"Deltas: {si_feature_archive.use_deltas}")
                train_logger.debug(f"Splices: {si_feature_archive.use_splices}")
                train_logger.debug(f"LDA: {si_feature_archive.lda_mat_file_name}")
                train_logger.debug(f"fMLLR: {si_feature_archive.transform_read_specifier}")
                train_logger.debug(f"\nAlignment path: {ali_path}")
                alignment_archive = AlignmentArchive(ali_path)
                accumulator.accumulate_stats(
                    feature_archive, si_feature_archive, alignment_archive, callback=self.callback
                )
                self.callback((accumulator.transition_accs, accumulator.gmm_accs))


class SatTrainer(TriphoneTrainer):
    """
    Speaker adapted trainer (SAT), inherits from TriphoneTrainer

    Parameters
    ----------
    subset : int
        Number of utterances to use, defaults to 10000
    num_leaves : int
        Number of states in the decision tree, defaults to 2500
    max_gaussians : int
        Number of gaussians in the decision tree, defaults to 15000
    power : float
        Exponent for number of gaussians according to occurrence counts, defaults to 0.2

    See Also
    --------
    :class:`~montreal_forced_aligner.acoustic_modeling.triphone.TriphoneTrainer`
        For acoustic model training parsing parameters

    Attributes
    ----------
    fmllr_iterations : list
        List of iterations to perform fMLLR calculation
    """

    def __init__(
        self,
        subset: int = 10000,
        num_leaves: int = 2500,
        max_gaussians: int = 15000,
        power: float = 0.2,
        boost_silence: float = 1.0,
        quick: bool = False,
        **kwargs,
    ):
        super().__init__(
            power=power,
            subset=subset,
            num_leaves=num_leaves,
            max_gaussians=max_gaussians,
            boost_silence=boost_silence,
            **kwargs,
        )
        self.fmllr_iterations = []
        self.quick = quick
        if self.quick:
            self.power = 0.2

    def acc_stats_two_feats_arguments(self) -> List[AccStatsTwoFeatsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.acoustic_modeling.sat.AccStatsTwoFeatsFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.acoustic_modeling.sat.AccStatsTwoFeatsArguments`]
            Arguments for processing
        """
        arguments = []
        for j in self.jobs:
            arguments.append(
                AccStatsTwoFeatsArguments(
                    j.id,
                    getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                    self.working_log_directory.joinpath(f"acc_stats_two_feats.{j.id}.log"),
                    self.working_directory,
                    self.model_path,
                )
            )
        return arguments

    def calc_fmllr(self) -> None:
        """Calculate fMLLR transforms for the current iteration"""
        self.worker.calc_fmllr(iteration=self.iteration)

    def compute_calculated_properties(self) -> None:
        """Generate realignment iterations, initial gaussians, and fMLLR iterations based on configuration"""
        super().compute_calculated_properties()
        self.fmllr_iterations = []
        if not self.quick:
            self.fmllr_iterations = [2, 4, 6, 12]
        else:
            self.realignment_iterations = [10, 15]
            self.fmllr_iterations = [2, 6, 12]
            self.final_gaussian_iteration = self.num_iterations - 5
            self.initial_gaussians = int(self.max_gaussians / 2)
            if self.initial_gaussians < self.num_leaves:
                self.initial_gaussians = self.num_leaves

    def _trainer_initialization(self) -> None:
        """Speaker adapted training initialization"""
        if self.initialized:
            self.uses_speaker_adaptation = True
            self.worker.uses_speaker_adaptation = True
            return
        if os.path.exists(os.path.join(self.previous_aligner.working_directory, "lda.mat")):
            shutil.copyfile(
                os.path.join(self.previous_aligner.working_directory, "lda.mat"),
                self.working_directory.joinpath("lda.mat"),
            )
        for j in self.jobs:
            for path in j.construct_path_dictionary(
                j.corpus.current_subset_directory, "trans", "scp"
            ).values():
                if path.exists():
                    break
            else:
                continue
            break
        else:
            self.uses_speaker_adaptation = False
            self.worker.uses_speaker_adaptation = False
            self.calc_fmllr()
        self.uses_speaker_adaptation = True
        self.worker.uses_speaker_adaptation = True
        self._setup_tree(init_from_previous=self.quick, initial_mix_up=self.quick)

        self.convert_alignments()

        self.compile_train_graphs()
        os.rename(self.model_path, self.next_model_path)

        self.iteration = 1
        parse_logs(self.working_log_directory)

    def finalize_training(self) -> None:
        """
        Finalize training and create a speaker independent model for initial alignment

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        try:
            self.create_align_model()
            self.uses_speaker_adaptation = True
            super().finalize_training()
            assert self.alignment_model_path.name == "final.alimdl"
            assert self.alignment_model_path.exists()
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs)
                e.update_log_file()
            raise

    def train_iteration(self) -> None:
        """
        Run a single training iteration
        """
        if os.path.exists(self.next_model_path):
            if self.iteration <= self.final_gaussian_iteration:
                self.increment_gaussians()
            self.iteration += 1
            return
        if self.iteration in self.realignment_iterations:
            self.align_iteration()
        if self.iteration in self.fmllr_iterations:
            self.calc_fmllr()

        self.acc_stats()

        if self.iteration <= self.final_gaussian_iteration:
            self.increment_gaussians()
        self.iteration += 1

    @property
    def alignment_model_path(self) -> Path:
        """Alignment model path"""
        path = self.model_path.with_suffix(".alimdl")
        if os.path.exists(path):
            return path
        return self.model_path

    def create_align_model(self) -> None:
        """
        Create alignment model for speaker-adapted training that will use speaker-independent
        features in later aligning.

        See Also
        --------
        :func:`~montreal_forced_aligner.acoustic_modeling.sat.AccStatsTwoFeatsFunction`
            Multiprocessing helper function for each job
        :meth:`.SatTrainer.acc_stats_two_feats_arguments`
            Job method for generating arguments for the helper function
        :kaldi_src:`gmm-est`
            Relevant Kaldi binary
        :kaldi_src:`gmm-sum-accs`
            Relevant Kaldi binary
        :kaldi_steps:`train_sat`
            Reference Kaldi script
        """
        logger.info("Creating alignment model for speaker-independent features...")
        begin = time.time()

        arguments = self.acc_stats_two_feats_arguments()

        transition_model, acoustic_model = read_gmm_model(self.model_path)
        transition_accs = DoubleVector()
        gmm_accs = AccumAmDiagGmm()
        transition_model.InitStats(transition_accs)
        gmm_accs.init(acoustic_model)
        for result in run_kaldi_function(
            AccStatsTwoFeatsFunction, arguments, total_count=self.num_current_utterances
        ):
            if isinstance(result, tuple):
                job_transition_accs, job_gmm_accs = result

                transition_accs.AddVec(1.0, job_transition_accs)
                gmm_accs.Add(1.0, job_gmm_accs)

        log_path = self.working_log_directory.joinpath("align_model_est.log")

        with kalpy_logger("kalpy.train", log_path):
            objf_impr, count = transition_model.mle_update(transition_accs)
            logger.debug(
                f"Transition model update: Overall {objf_impr/count} "
                f"log-like improvement per frame over {count} frames."
            )
            objf_impr, count = acoustic_model.mle_update(
                gmm_accs,
                mixup=self.current_gaussians,
                power=self.power,
                remove_low_count_gaussians=False,
            )
            logger.debug(
                f"GMM update: Overall {objf_impr/count} "
                f"objective function improvement per frame over {count} frames."
            )
            tot_like = gmm_accs.TotLogLike()
            tot_t = gmm_accs.TotCount()
            logger.debug(
                f"Average Likelihood per frame for iteration = {tot_like/tot_t} "
                f"over {tot_t} frames."
            )
            write_gmm_model(
                self.model_path.with_suffix(".alimdl"), transition_model, acoustic_model
            )

        logger.debug(f"Alignment model creation took {time.time() - begin:.3f} seconds")
