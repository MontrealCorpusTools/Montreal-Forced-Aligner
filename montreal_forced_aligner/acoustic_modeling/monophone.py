"""Class definitions for Monophone trainer"""
from __future__ import annotations

import logging
import typing
from pathlib import Path

from _kalpy.gmm import AccumAmDiagGmm, gmm_align_equal, gmm_init_mono
from _kalpy.matrix import DoubleVector
from _kalpy.util import Int32VectorWriter
from kalpy.decoder.data import FstArchive
from kalpy.feat.data import FeatureArchive
from kalpy.gmm.train import GmmStatsAccumulator
from kalpy.gmm.utils import read_gmm_model, read_topology, read_tree, write_gmm_model
from kalpy.utils import generate_write_specifier, kalpy_logger
from sqlalchemy.orm import joinedload, subqueryload

from montreal_forced_aligner import config
from montreal_forced_aligner.abc import KaldiFunction, MetaDict
from montreal_forced_aligner.acoustic_modeling.base import AcousticModelTrainingMixin
from montreal_forced_aligner.data import MfaArguments
from montreal_forced_aligner.db import Job
from montreal_forced_aligner.utils import run_kaldi_function, thread_logger

__all__ = ["MonophoneTrainer", "MonoAlignEqualFunction", "MonoAlignEqualArguments"]

logger = logging.getLogger("mfa")


class MonoAlignEqualArguments(MfaArguments):
    """Arguments for :func:`~montreal_forced_aligner.acoustic_modeling.monophone.MonoAlignEqualFunction`"""

    working_directory: Path
    model_path: Path


class MonoAlignEqualFunction(KaldiFunction):
    """
    Multiprocessing function for initializing monophone alignments

    See Also
    --------
    :meth:`.MonophoneTrainer.mono_align_equal`
        Main function that calls this function in parallel
    :meth:`.MonophoneTrainer.mono_align_equal_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`align-equal-compiled`
        Relevant Kaldi binary
    :kaldi_src:`gmm-acc-stats-ali`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.acoustic_modeling.monophone.MonoAlignEqualArguments`
        Arguments for the function
    """

    def __init__(self, args: MonoAlignEqualArguments):
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
            num_done = 0
            num_error = 0
            tot_like = 0.0
            tot_t = 0.0
            for d in job.training_dictionaries:
                dict_id = d.id
                train_logger.debug(f"Aligning for dictionary {d.name} ({d.id})")
                train_logger.debug(f"Aligning with model: {self.model_path}")
                fst_path = job.construct_path(self.working_directory, "fsts", "ark", dict_id)
                train_logger.debug(f"Training graph archive: {fst_path}")
                accumulator = GmmStatsAccumulator(self.model_path)
                feat_path = job.construct_path(
                    job.corpus.current_subset_directory, "feats", "scp", dictionary_id=dict_id
                )
                train_logger.debug(f"Feature path: {feat_path}")
                feature_archive = FeatureArchive(
                    feat_path,
                    deltas=True,
                )
                training_graph_archive = FstArchive(fst_path)
                ali_path = job.construct_path(self.working_directory, "ali", "ark", dict_id)
                write_specifier = generate_write_specifier(ali_path, write_scp=False)
                writer = Int32VectorWriter(write_specifier)
                for utt_id, decode_fst in training_graph_archive:
                    train_logger.debug(f"Aligning {utt_id}")
                    feats = feature_archive[utt_id]
                    if feats.NumRows() == 0:
                        train_logger.warning(f"Zero-length utterance: {utt_id}")
                        num_error += 1
                        continue
                    if decode_fst.Start() == -1:
                        train_logger.warning(f"Empty decoding graph for {utt_id}")
                        num_error += 1
                        continue
                    alignment, words = gmm_align_equal(decode_fst, feats)
                    if alignment is None or len(alignment) == 0:
                        train_logger.warning(f"AlignEqual: did not align utterance {utt_id}")
                        num_error += 1
                        continue
                    writer.Write(utt_id, alignment)
                    tot_like_this_file = accumulator.gmm_accs.acc_stats(
                        accumulator.acoustic_model,
                        accumulator.transition_model,
                        alignment,
                        feats,
                    )
                    accumulator.transition_model.acc_stats(alignment, accumulator.transition_accs)

                    num_done += 1
                    tot_like += tot_like_this_file
                    tot_t += len(alignment)
                    if num_done % 50 == 0:
                        train_logger.info(
                            f"Processed {num_done} utterances; for utterance "
                            f"{utt_id} avg. like is "
                            f"{tot_like_this_file / len(alignment)} "
                            f"over {len(alignment)} frames."
                        )
                    self.callback(utt_id)
                writer.Close()
                self.callback((accumulator.transition_accs, accumulator.gmm_accs))
                train_logger.info(f"Done {num_done} utterances, errors on {num_error} utterances.")
                if tot_t:
                    train_logger.info(
                        f"Overall avg like per frame (Gaussian only) = {tot_like/tot_t} over {tot_t} frames."
                    )


class MonophoneTrainer(AcousticModelTrainingMixin):
    """
    Configuration class for monophone training

    Attributes
    ----------
    subset : int
        Number of utterances to use, defaults to 2000
    initial_gaussians : int
        Number of gaussians to begin training, defaults to 135
    max_gaussians : int
        Total number of gaussians, defaults to 1000
    power : float
        Exponent for number of gaussians according to occurrence counts, defaults to 0.25

    See Also
    --------
    :class:`~montreal_forced_aligner.acoustic_modeling.base.AcousticModelTrainingMixin`
        For acoustic model training parsing parameters
    """

    def __init__(
        self,
        subset: int = 2000,
        initial_gaussians: int = 135,
        initial_beam: int = 6,
        max_gaussians: int = 1000,
        power: float = 0.25,
        boost_silence: float = 1.25,
        **kwargs,
    ):
        super().__init__(
            power=power,
            subset=subset,
            initial_gaussians=initial_gaussians,
            max_gaussians=max_gaussians,
            boost_silence=boost_silence,
            **kwargs,
        )
        self.subset = subset
        self.initial_beam = initial_beam
        self.last_gaussian_increase_iteration = 0

    def mono_align_equal_arguments(self) -> typing.List[MonoAlignEqualArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.acoustic_modeling.monophone.MonoAlignEqualFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.acoustic_modeling.monophone.MonoAlignEqualArguments`]
            Arguments for processing
        """
        return [
            MonoAlignEqualArguments(
                j.id,
                getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                self.working_log_directory.joinpath(f"mono_align_equal.{j.id}.log"),
                self.working_directory,
                self.model_path,
            )
            for j in self.jobs
        ]

    def compute_calculated_properties(self) -> None:
        """Generate realignment iterations and initial gaussians based on configuration"""
        self.final_gaussian_iteration = self.num_iterations - 10
        self.realignment_iterations = [0]
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

    @property
    def align_options(self) -> MetaDict:
        """Alignment parameters"""
        options = super().align_options
        if self.iteration == 1:
            options["beam"] = self.initial_beam
        return options

    def mono_align_equal(self) -> None:
        """
        Multiprocessing function that creates equal alignments for base monophone training.

        See Also
        --------
        :func:`~montreal_forced_aligner.acoustic_modeling.monophone.MonoAlignEqualFunction`
            Multiprocessing helper function for each job
        :meth:`.MonophoneTrainer.mono_align_equal_arguments`
            Job method for generating arguments for the helper function
        :kaldi_src:`gmm-sum-accs`
            Relevant Kaldi binary
        :kaldi_src:`gmm-est`
            Relevant Kaldi binary
        :kaldi_steps:`train_mono`
            Reference Kaldi script
        """

        logger.info("Generating initial alignments...")
        arguments = self.mono_align_equal_arguments()
        transition_model, acoustic_model = read_gmm_model(self.model_path)
        transition_accs = DoubleVector()
        gmm_accs = AccumAmDiagGmm()
        transition_model.InitStats(transition_accs)
        gmm_accs.init(acoustic_model)
        log_path = self.working_log_directory.joinpath("mono_align_equal.log")
        with kalpy_logger("kalpy.train", log_path):
            for result in run_kaldi_function(
                MonoAlignEqualFunction, arguments, total_count=self.num_current_utterances
            ):
                if isinstance(result, tuple):
                    job_transition_accs, job_gmm_accs = result

                    transition_accs.AddVec(1.0, job_transition_accs)
                    gmm_accs.Add(1.0, job_gmm_accs)

        log_path = self.working_log_directory.joinpath("update.0.log")
        with kalpy_logger("kalpy.train", log_path):
            objf_impr, count = transition_model.mle_update(transition_accs)
            logger.debug(
                f"Transition model update: Overall {objf_impr/count} "
                f"log-like improvement per frame over {count} frames."
            )
            objf_impr, count = acoustic_model.mle_update(
                gmm_accs,
                min_gaussian_occupancy=3.0,
                mixup=self.current_gaussians,
                power=self.power,
            )
            logger.debug(
                f"GMM update: Overall {objf_impr/count} "
                f"objective function improvement per frame over {count} frames."
            )
            tot_like = gmm_accs.TotLogLike()
            tot_t = gmm_accs.TotCount()
            logger.debug(
                f"Average Likelihood per frame for iteration {self.iteration} = {tot_like/tot_t} "
                f"over {tot_t} frames."
            )
            write_gmm_model(str(self.next_model_path), transition_model, acoustic_model)

    def _trainer_initialization(self) -> None:
        """Monophone training initialization"""
        if self.initialized:
            return
        self.iteration = 0
        tree_path = self.working_directory.joinpath("tree")
        init_log_path = self.working_log_directory.joinpath("init.log")
        job = self.jobs[0]
        feats = []
        with kalpy_logger("kalpy.train", init_log_path) as train_logger:
            dict_index = 0
            while len(feats) < 10:
                try:
                    dict_id = job.dictionary_ids[dict_index]
                except IndexError:
                    break
                feature_archive = job.construct_feature_archive(self.working_directory, dict_id)
                for i, (_, mat) in enumerate(feature_archive):
                    if i > 10:
                        break
                    feats.append(mat)
                dict_index += 1
            if not feats:
                raise Exception("Could not initialize monophone model due to lack of features")
            shared_phones = self.worker.shared_phones_set_symbols()
            topo = read_topology(self.worker.topo_path)
            gmm_init_mono(topo, feats, shared_phones, str(self.model_path), str(tree_path))
            transition_model, acoustic_model = read_gmm_model(self.model_path)
            num_gauss = acoustic_model.NumGauss()
            tree = read_tree(tree_path)
            train_logger.debug(
                f"Initialized monophone model with {num_gauss} gaussians, "
                f"{acoustic_model.NumPdfs()} pdfs"
            )
            train_logger.debug(
                f"Transition model with {transition_model.NumTransitionIds()} transition ids, "
                f"{transition_model.NumPdfs()} pdfs"
            )
            train_logger.debug(f"Tree with {tree.NumPdfs()}")
        self.initial_gaussians = num_gauss
        self.current_gaussians = num_gauss
        self.compile_train_graphs()
        self.mono_align_equal()
