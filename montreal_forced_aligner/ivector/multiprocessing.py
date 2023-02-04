"""Multiprocessing functions for training ivector extractors"""
from __future__ import annotations

import os
import re
import subprocess
import typing

from sqlalchemy.orm import Session, joinedload

from montreal_forced_aligner.abc import MetaDict
from montreal_forced_aligner.data import MfaArguments
from montreal_forced_aligner.db import Job
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.utils import KaldiFunction, thirdparty_binary

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

    feature_options: MetaDict
    ivector_options: MetaDict
    dubm_model: str
    gselect_path: str


class AccGlobalStatsArguments(MfaArguments):
    """Arguments for :func:`~montreal_forced_aligner.ivector.trainer.AccGlobalStatsFunction`"""

    feature_options: MetaDict
    ivector_options: MetaDict
    gselect_path: str
    acc_path: str
    dubm_model: str


class GaussToPostArguments(MfaArguments):
    """Arguments for :func:`~montreal_forced_aligner.ivector.trainer.GaussToPostFunction`"""

    feature_options: MetaDict
    ivector_options: MetaDict
    post_path: str
    dubm_model: str


class AccIvectorStatsArguments(MfaArguments):
    """Arguments for :func:`~montreal_forced_aligner.ivector.trainer.AccIvectorStatsFunction`"""

    feature_options: MetaDict
    ivector_options: MetaDict
    ie_path: str
    post_path: str
    acc_path: str


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

    progress_pattern = re.compile(r"^LOG.*For (?P<done_count>\d+)'th.*")

    def __init__(self, args: GmmGselectArguments):
        super().__init__(args)
        self.feature_options = args.feature_options
        self.ivector_options = args.ivector_options
        self.dubm_model = args.dubm_model
        self.gselect_path = args.gselect_path

    def _run(self) -> typing.Generator[None]:
        """Run the function"""
        if os.path.exists(self.gselect_path):
            return
        with Session(self.db_engine) as session, mfa_open(self.log_path, "w") as log_file:
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True))
                .filter(Job.id == self.job_name)
                .first()
            )
            current_done_count = 0
            feature_string = job.construct_online_feature_proc_string()

            gselect_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-gselect"),
                    f"--n={self.ivector_options['num_gselect']}",
                    self.dubm_model,
                    feature_string,
                    f"ark:{self.gselect_path}",
                ],
                stderr=subprocess.PIPE,
                env=os.environ,
                encoding="utf8",
            )
            for line in gselect_proc.stderr:
                log_file.write(line)
                m = self.progress_pattern.match(line)
                if m:
                    new_done_count = int(m.group("done_count"))
                    yield new_done_count - current_done_count
                    current_done_count = new_done_count
            self.check_call(gselect_proc)


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

    progress_pattern = re.compile(
        r"^VLOG.*Processed utterance (?P<utterance>.*), average likelihood.*$"
    )

    def __init__(self, args: GaussToPostArguments):
        super().__init__(args)
        self.feature_options = args.feature_options
        self.ivector_options = args.ivector_options
        self.dubm_model = args.dubm_model
        self.post_path = args.post_path

    def _run(self) -> typing.Generator[None]:
        """Run the function"""
        if os.path.exists(self.post_path):
            return
        modified_posterior_scale = (
            self.ivector_options["posterior_scale"] * self.ivector_options["subsample"]
        )
        with Session(self.db_engine) as session, mfa_open(self.log_path, "w") as log_file:
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True))
                .filter(Job.id == self.job_name)
                .first()
            )
            feature_string = job.construct_online_feature_proc_string()
            gmm_global_get_post_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-global-get-post"),
                    "--verbose=2",
                    f"--n={self.ivector_options['num_gselect']}",
                    f"--min-post={self.ivector_options['min_post']}",
                    self.dubm_model,
                    feature_string,
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ,
            )
            scale_post_proc = subprocess.Popen(
                [
                    thirdparty_binary("scale-post"),
                    "ark,s,cs:-",
                    str(modified_posterior_scale),
                    f"ark:{self.post_path}",
                ],
                stdin=gmm_global_get_post_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            for line in gmm_global_get_post_proc.stderr:
                line = line.decode("utf8")
                log_file.write(line)
                log_file.flush()
                m = self.progress_pattern.match(line)
                if m:
                    utterance = int(m.group("utterance").split("-")[-1])
                    yield utterance
            self.check_call(scale_post_proc)


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

    progress_pattern = re.compile(r"^VLOG.*File '(?P<file>.*)': Average likelihood =.*$")

    def __init__(self, args: AccGlobalStatsArguments):
        super().__init__(args)
        self.feature_options = args.feature_options
        self.ivector_options = args.ivector_options
        self.dubm_model = args.dubm_model
        self.gselect_path = args.gselect_path
        self.acc_path = args.acc_path

    def _run(self) -> typing.Generator[None]:
        """Run the function"""
        with Session(self.db_engine) as session, mfa_open(self.log_path, "w") as log_file:
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True))
                .filter(Job.id == self.job_name)
                .first()
            )
            feature_string = job.construct_online_feature_proc_string()
            command = [
                thirdparty_binary("gmm-global-acc-stats"),
                "--verbose=2",
                f"--gselect=ark,s,cs:{self.gselect_path}",
                self.dubm_model,
                feature_string,
                self.acc_path,
            ]
            gmm_global_acc_proc = subprocess.Popen(
                command,
                stderr=subprocess.PIPE,
                env=os.environ,
                encoding="utf8",
            )
            for line in gmm_global_acc_proc.stderr:
                log_file.write(line)
                log_file.flush()
                m = self.progress_pattern.match(line)
                if m:
                    utt_id = int(m.group("file").split("-")[-1])
                    yield utt_id
            self.check_call(gmm_global_acc_proc)


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

    progress_pattern = re.compile(r"VLOG.* Per frame, auxf is: weight.*")

    def __init__(self, args: AccIvectorStatsArguments):
        super().__init__(args)
        self.feature_options = args.feature_options
        self.ivector_options = args.ivector_options
        self.ie_path = args.ie_path
        self.post_path = args.post_path
        self.acc_path = args.acc_path

    def _run(self) -> typing.Generator[None]:
        """Run the function"""
        with Session(self.db_engine) as session, mfa_open(self.log_path, "w") as log_file:
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True))
                .filter(Job.id == self.job_name)
                .first()
            )
            feature_string = job.construct_online_feature_proc_string()
            acc_stats_proc = subprocess.Popen(
                [
                    thirdparty_binary("ivector-extractor-acc-stats"),
                    "--verbose=4",
                    self.ie_path,
                    feature_string,
                    f"ark,s,cs:{self.post_path}",
                    self.acc_path,
                ],
                stderr=subprocess.PIPE,
                env=os.environ,
                encoding="utf8",
            )
            for line in acc_stats_proc.stderr:
                m = self.progress_pattern.match(line)
                if m:
                    yield 1
                    continue
                elif "VLOG" in line:
                    continue
                log_file.write(line)
                log_file.flush()
            self.check_call(acc_stats_proc)
