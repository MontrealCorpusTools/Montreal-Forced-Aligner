"""Multiprocessing functions for training ivector extractors"""
from __future__ import annotations

import os
from pathlib import Path

from _kalpy.gmm import DiagGmm
from _kalpy.hmm import PosteriorWriter, RandomAccessPosteriorReader, ScalePosterior
from _kalpy.util import Int32VectorVectorWriter
from kalpy.ivector.data import GselectArchive
from kalpy.ivector.train import GlobalGmmStatsAccumulator, IvectorExtractorStatsAccumulator
from kalpy.utils import generate_read_specifier, generate_write_specifier, read_kaldi_object
from sqlalchemy.orm import joinedload

from montreal_forced_aligner.abc import KaldiFunction, MetaDict
from montreal_forced_aligner.data import MfaArguments
from montreal_forced_aligner.db import Job
from montreal_forced_aligner.utils import thread_logger

__all__ = [
    "GmmGselectFunction",
    "GmmGselectArguments",
    "GaussToPostFunction",
    "GaussToPostArguments",
    "AccGlobalStatsFunction",
    "AccGlobalStatsArguments",
    "AccIvectorStatsFunction",
    "AccIvectorStatsArguments",
]


class GmmGselectArguments(MfaArguments):
    """Arguments for :func:`~montreal_forced_aligner.ivector.trainer.GmmGselectFunction`"""

    working_directory: Path
    dubm_model: Path
    ivector_options: MetaDict


class AccGlobalStatsArguments(MfaArguments):
    """Arguments for :func:`~montreal_forced_aligner.ivector.trainer.AccGlobalStatsFunction`"""

    working_directory: Path
    dubm_model: Path
    ivector_options: MetaDict


class GaussToPostArguments(MfaArguments):
    """Arguments for :func:`~montreal_forced_aligner.ivector.trainer.GaussToPostFunction`"""

    working_directory: Path
    dubm_model: Path
    ivector_options: MetaDict


class AccIvectorStatsArguments(MfaArguments):
    """Arguments for :func:`~montreal_forced_aligner.ivector.trainer.AccIvectorStatsFunction`"""

    working_directory: Path
    ie_path: Path
    ivector_options: MetaDict


class GmmGselectFunction(KaldiFunction):
    """
    Multiprocessing function for selecting GMM indices.

    See Also
    --------
    :meth:`.DubmTrainer.gmm_gselect`
        Main function that calls this function in parallel
    :meth:`.DubmTrainer.gmm_gselect_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`subsample-feats`
        Relevant Kaldi binary
    :kaldi_src:`gmm-gselect`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.ivector.trainer.GmmGselectArguments`
        Arguments for the function
    """

    def __init__(self, args: GmmGselectArguments):
        super().__init__(args)
        self.working_directory = args.working_directory
        self.dubm_model = args.dubm_model
        self.ivector_options = args.ivector_options

    def _run(self) -> None:
        """Run the function"""
        with self.session() as session, thread_logger(
            "kalpy.ivector", self.log_path, job_name=self.job_name
        ) as ivector_logger:
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True))
                .filter(Job.id == self.job_name)
                .first()
            )
            gselect_path = job.construct_path(self.working_directory, "gselect", "ark")
            if os.path.exists(gselect_path):
                return
            feature_archive = job.construct_feature_archive(
                job.corpus.split_directory,
                subsample_n=self.ivector_options["subsample"],
                use_sliding_cmvn=self.ivector_options["uses_cmvn"],
            )
            gmm = read_kaldi_object(DiagGmm, self.dubm_model)
            gselect_writer = Int32VectorVectorWriter(generate_write_specifier(gselect_path))
            num_done = 0
            num_skipped = 0
            tot_like = 0.0
            tot_t = 0.0
            for utt_id, feats in feature_archive:
                tot_t_this_file = feats.NumRows()
                if tot_t_this_file == 0:
                    ivector_logger.warning(f"Skipping {utt_id} due to zero-length features.")
                    num_skipped += 1
                    continue
                gselect, tot_like_this_file = gmm.gaussian_selection(
                    feats, self.ivector_options["num_gselect"]
                )
                gselect_writer.Write(utt_id, gselect)
                num_done += 1
                tot_like += tot_like_this_file
                tot_t += tot_t_this_file
                if num_done % 10 == 0:
                    self.callback(10)
                    ivector_logger.info(
                        f"For {num_done}'th utterance, "
                        f"average UBM log-likelihood over {tot_t_this_file} frames "
                        f"is {tot_like_this_file/tot_t_this_file}."
                    )
            gselect_writer.Close()
            ivector_logger.info(
                f"Done {num_done} utterances, skipped {num_skipped}, "
                f"average UBM log-likelihood over {tot_t} frames is {tot_like/tot_t}."
            )


class GaussToPostFunction(KaldiFunction):
    """
    Multiprocessing function to get posteriors during UBM training.

    See Also
    --------
    :meth:`.IvectorTrainer.gauss_to_post`
        Main function that calls this function in parallel
    :meth:`.IvectorTrainer.gauss_to_post_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`subsample-feats`
        Relevant Kaldi binary
    :kaldi_src:`gmm-global-get-post`
        Relevant Kaldi binary
    :kaldi_src:`scale-post`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.ivector.trainer.GaussToPostArguments`
        Arguments for the function
    """

    def __init__(self, args: GaussToPostArguments):
        super().__init__(args)
        self.working_directory = args.working_directory
        self.dubm_model = args.dubm_model
        self.ivector_options = args.ivector_options

    def _run(self) -> None:
        """Run the function"""
        modified_posterior_scale = (
            self.ivector_options["posterior_scale"] * self.ivector_options["subsample"]
        )
        with self.session() as session, thread_logger(
            "kalpy.ivector", self.log_path, job_name=self.job_name
        ) as ivector_logger:
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True))
                .filter(Job.id == self.job_name)
                .first()
            )
            post_path = job.construct_path(self.working_directory, "post", "ark")
            if os.path.exists(post_path):
                return
            feature_archive = job.construct_feature_archive(
                job.corpus.split_directory,
                subsample_n=self.ivector_options["subsample"],
                use_sliding_cmvn=self.ivector_options["uses_cmvn"],
            )
            gmm: DiagGmm = read_kaldi_object(DiagGmm, self.dubm_model)
            num_done = 0
            num_skipped = 0
            tot_like = 0.0
            tot_t = 0.0
            post_writer = PosteriorWriter(generate_write_specifier(post_path))
            for utt_id, feats in feature_archive:
                tot_t_this_file = feats.NumRows()
                if tot_t_this_file == 0:
                    ivector_logger.warning(f"Skipping {utt_id} due to zero-length features.")
                    num_skipped += 1
                    continue
                if feats.NumCols() != gmm.Dim():
                    ivector_logger.warning(
                        f"Dimension mismatch for utterance {utt_id}: "
                        f"got {feats.NumCols()}, expected {gmm.Dim()}"
                    )
                    num_skipped += 1
                    continue
                post, tot_like_this_file = gmm.generate_post(
                    feats,
                    num_post=self.ivector_options["num_gselect"],
                    min_post=self.ivector_options["min_post"],
                )
                ScalePosterior(modified_posterior_scale, post)
                tot_like += tot_like_this_file
                tot_t += tot_t_this_file
                post_writer.Write(utt_id, post)
                num_done += 1
                if num_done % 10 == 0:
                    self.callback(10)
            post_writer.Close()

            ivector_logger.info(
                f"Done {num_done} utterances, skipped {num_skipped}, "
                f"average UBM log-likelihood over {tot_t} frames is {tot_like/tot_t}."
            )


class AccGlobalStatsFunction(KaldiFunction):
    """
    Multiprocessing function for accumulating global model stats.

    See Also
    --------
    :meth:`.DubmTrainer.acc_global_stats`
        Main function that calls this function in parallel
    :meth:`.DubmTrainer.acc_global_stats_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`subsample-feats`
        Relevant Kaldi binary
    :kaldi_src:`gmm-global-acc-stats`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.ivector.trainer.AccGlobalStatsArguments`
        Arguments for the function
    """

    def __init__(self, args: AccGlobalStatsArguments):
        super().__init__(args)
        self.ivector_options = args.ivector_options
        self.dubm_model = args.dubm_model
        self.working_directory = args.working_directory

    def _run(self):
        """Run the function"""
        with self.session() as session, thread_logger(
            "kalpy.ivector", self.log_path, job_name=self.job_name
        ):
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True))
                .filter(Job.id == self.job_name)
                .first()
            )
            feature_archive = job.construct_feature_archive(
                job.corpus.split_directory,
                subsample_n=self.ivector_options["subsample"],
                use_sliding_cmvn=self.ivector_options["uses_cmvn"],
            )
            gselect_path = job.construct_path(self.working_directory, "gselect", "ark")
            gselect_archive = GselectArchive(gselect_path)
            accumulator = GlobalGmmStatsAccumulator(self.dubm_model)
            accumulator.accumulate_stats(feature_archive, gselect_archive, callback=self.callback)
            self.callback(accumulator.gmm_accs)


class AccIvectorStatsFunction(KaldiFunction):
    """
    Multiprocessing function that accumulates stats for ivector training.

    See Also
    --------
    :meth:`.IvectorTrainer.acc_ivector_stats`
        Main function that calls this function in parallel
    :meth:`.IvectorTrainer.acc_ivector_stats_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`subsample-feats`
        Relevant Kaldi binary
    :kaldi_src:`ivector-extractor-acc-stats`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.ivector.trainer.AccIvectorStatsArguments`
        Arguments for the function
    """

    def __init__(self, args: AccIvectorStatsArguments):
        super().__init__(args)
        self.working_directory = args.working_directory
        self.ie_path = args.ie_path
        self.ivector_options = args.ivector_options

    def _run(self):
        """Run the function"""
        with self.session() as session, thread_logger(
            "kalpy.ivector", self.log_path, job_name=self.job_name
        ):
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True))
                .filter(Job.id == self.job_name)
                .first()
            )
            feature_archive = job.construct_feature_archive(
                job.corpus.split_directory,
                subsample_n=self.ivector_options["subsample"],
                use_sliding_cmvn=self.ivector_options["uses_cmvn"],
            )
            post_path = job.construct_path(self.working_directory, "post", "ark")
            post_reader = RandomAccessPosteriorReader(generate_read_specifier(post_path))
            accumulator = IvectorExtractorStatsAccumulator(self.ie_path)
            accumulator.accumulate_stats(feature_archive, post_reader, callback=self.callback)
            self.callback(accumulator.ivector_stats)
            post_reader.Close()
