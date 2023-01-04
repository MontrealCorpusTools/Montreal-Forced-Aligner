"""Class definitions for LDA trainer"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import typing
from queue import Empty
from typing import TYPE_CHECKING, Dict, List

import tqdm

from montreal_forced_aligner.abc import KaldiFunction
from montreal_forced_aligner.acoustic_modeling.triphone import TriphoneTrainer
from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.data import MfaArguments
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.utils import (
    KaldiProcessWorker,
    Stopped,
    parse_logs,
    thirdparty_binary,
)

if TYPE_CHECKING:
    from montreal_forced_aligner.abc import MetaDict


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

    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ali_paths: Dict[str, str]
    model_path: str
    lda_options: MetaDict
    acc_paths: Dict[str, str]


class CalcLdaMlltArguments(MfaArguments):
    """Arguments for :func:`~montreal_forced_aligner.acoustic_modeling.lda.CalcLdaMlltFunction`"""

    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ali_paths: Dict[str, str]
    model_path: str
    lda_options: MetaDict
    macc_paths: Dict[str, str]


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

    progress_pattern = re.compile(r"^LOG.*Done (?P<done>\d+) files, failed for (?P<failed>\d+)$")

    def __init__(self, args: LdaAccStatsArguments):
        super().__init__(args)
        self.dictionaries = args.dictionaries
        self.feature_strings = args.feature_strings
        self.ali_paths = args.ali_paths
        self.model_path = args.model_path
        self.acc_paths = args.acc_paths
        self.lda_options = args.lda_options

    def _run(self) -> typing.Generator[typing.Tuple[int, int]]:
        """Run the function"""
        with mfa_open(self.log_path, "w") as log_file:
            for dict_id in self.dictionaries:
                ali_path = self.ali_paths[dict_id]
                feature_string = self.feature_strings[dict_id]
                acc_path = self.acc_paths[dict_id]
                ali_to_post_proc = subprocess.Popen(
                    [thirdparty_binary("ali-to-post"), f"ark:{ali_path}", "ark:-"],
                    stderr=log_file,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                )
                weight_silence_post_proc = subprocess.Popen(
                    [
                        thirdparty_binary("weight-silence-post"),
                        "0.0",
                        self.lda_options["silence_csl"],
                        self.model_path,
                        "ark:-",
                        "ark:-",
                    ],
                    stdin=ali_to_post_proc.stdout,
                    stderr=log_file,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                )
                acc_lda_post_proc = subprocess.Popen(
                    [
                        thirdparty_binary("acc-lda"),
                        f"--rand-prune={self.lda_options['random_prune']}",
                        self.model_path,
                        feature_string,
                        "ark,s,cs:-",
                        acc_path,
                    ],
                    stdin=weight_silence_post_proc.stdout,
                    stderr=subprocess.PIPE,
                    encoding="utf8",
                    env=os.environ,
                )
                for line in acc_lda_post_proc.stderr:
                    log_file.write(line)
                    m = self.progress_pattern.match(line.strip())
                    if m:
                        yield int(m.group("done")), int(m.group("failed"))
                self.check_call(acc_lda_post_proc)


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

    progress_pattern = re.compile(r"^LOG.*Average like for this file.*$")

    def __init__(self, args: CalcLdaMlltArguments):
        super().__init__(args)
        self.dictionaries = args.dictionaries
        self.feature_strings = args.feature_strings
        self.ali_paths = args.ali_paths
        self.model_path = args.model_path
        self.macc_paths = args.macc_paths
        self.lda_options = args.lda_options

    def _run(self) -> typing.Generator[int]:
        """Run the function"""
        # Estimating MLLT
        with mfa_open(self.log_path, "w") as log_file:
            for dict_id in self.dictionaries:
                ali_path = self.ali_paths[dict_id]
                feature_string = self.feature_strings[dict_id]
                macc_path = self.macc_paths[dict_id]
                post_proc = subprocess.Popen(
                    [thirdparty_binary("ali-to-post"), f"ark:{ali_path}", "ark:-"],
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=os.environ,
                )

                weight_proc = subprocess.Popen(
                    [
                        thirdparty_binary("weight-silence-post"),
                        "0.0",
                        self.lda_options["silence_csl"],
                        self.model_path,
                        "ark:-",
                        "ark:-",
                    ],
                    stdin=post_proc.stdout,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=os.environ,
                )
                acc_proc = subprocess.Popen(
                    [
                        thirdparty_binary("gmm-acc-mllt"),
                        f"--rand-prune={self.lda_options['random_prune']}",
                        self.model_path,
                        feature_string,
                        "ark,s,cs:-",
                        macc_path,
                    ],
                    stdin=weight_proc.stdout,
                    stderr=subprocess.PIPE,
                    encoding="utf8",
                    env=os.environ,
                )
                for line in acc_proc.stderr:
                    log_file.write(line)
                    m = self.progress_pattern.match(line.strip())
                    if m:
                        yield 1
                self.check_call(acc_proc)


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
        random_prune=4.0,
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

    def lda_acc_stats_arguments(self) -> List[LdaAccStatsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.acoustic_modeling.lda.LdaAccStatsFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.acoustic_modeling.lda.LdaAccStatsArguments`]
            Arguments for processing
        """
        arguments = []
        for j in self.jobs:
            feat_strings = {}
            for d_id in j.dictionary_ids:
                feat_strings[d_id] = j.construct_feature_proc_string(
                    self.working_directory,
                    d_id,
                    self.feature_options["uses_splices"],
                    self.feature_options["splice_left_context"],
                    self.feature_options["splice_right_context"],
                    self.feature_options["uses_speaker_adaptation"],
                )
            arguments.append(
                LdaAccStatsArguments(
                    j.id,
                    getattr(self, "db_string", ""),
                    os.path.join(self.working_log_directory, f"lda_acc_stats.{j.id}.log"),
                    j.dictionary_ids,
                    feat_strings,
                    j.construct_path_dictionary(
                        self.previous_aligner.working_directory, "ali", "ark"
                    ),
                    self.previous_aligner.alignment_model_path,
                    self.lda_options,
                    j.construct_path_dictionary(self.working_directory, "lda", "acc"),
                )
            )
        return arguments

    def calc_lda_mllt_arguments(self) -> List[CalcLdaMlltArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.acoustic_modeling.lda.CalcLdaMlltFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.acoustic_modeling.lda.CalcLdaMlltArguments`]
            Arguments for processing
        """
        arguments = []
        for j in self.jobs:
            feat_strings = {}
            for d_id in j.dictionary_ids:
                feat_strings[d_id] = j.construct_feature_proc_string(
                    self.working_directory,
                    d_id,
                    self.feature_options["uses_splices"],
                    self.feature_options["splice_left_context"],
                    self.feature_options["splice_right_context"],
                    self.feature_options["uses_speaker_adaptation"],
                )
            arguments.append(
                CalcLdaMlltArguments(
                    j.id,
                    getattr(self, "db_string", ""),
                    os.path.join(
                        self.working_log_directory, f"lda_mllt.{self.iteration}.{j.id}.log"
                    ),
                    j.dictionary_ids,
                    feat_strings,
                    j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                    self.model_path,
                    self.lda_options,
                    j.construct_path_dictionary(self.working_directory, "lda", "macc"),
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
            "silence_csl": self.silence_csl,
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
        worker_lda_path = os.path.join(self.worker.working_directory, "lda.mat")
        lda_path = os.path.join(self.working_directory, "lda.mat")
        if os.path.exists(worker_lda_path):
            os.remove(worker_lda_path)
        arguments = self.lda_acc_stats_arguments()
        with tqdm.tqdm(total=self.num_current_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
            if GLOBAL_CONFIG.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(arguments):
                    function = LdaAccStatsFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if isinstance(result, Exception):
                            error_dict[getattr(result, "job_name", 0)] = result
                            continue
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    done, errors = result
                    pbar.update(done + errors)
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in arguments:
                    function = LdaAccStatsFunction(args)
                    for done, errors in function.run():
                        pbar.update(done + errors)

        log_path = os.path.join(self.working_log_directory, "lda_est.log")
        acc_list = []
        for x in arguments:
            acc_list.extend(x.acc_paths.values())
        with mfa_open(log_path, "w") as log_file:
            est_lda_proc = subprocess.Popen(
                [
                    thirdparty_binary("est-lda"),
                    f"--dim={self.lda_dimension}",
                    lda_path,
                ]
                + acc_list,
                stderr=log_file,
                env=os.environ,
            )
            est_lda_proc.communicate()
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
        self.tree_stats()
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
        with tqdm.tqdm(total=self.num_current_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
            if GLOBAL_CONFIG.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(arguments):
                    function = CalcLdaMlltFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if isinstance(result, Exception):
                            error_dict[getattr(result, "job_name", 0)] = result
                            continue
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    pbar.update(1)
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in arguments:
                    function = CalcLdaMlltFunction(args)
                    for _ in function.run():
                        pbar.update(1)

        log_path = os.path.join(
            self.working_log_directory, f"transform_means.{self.iteration}.log"
        )
        previous_mat_path = os.path.join(self.working_directory, "lda.mat")
        new_mat_path = os.path.join(self.working_directory, "lda_new.mat")
        composed_path = os.path.join(self.working_directory, "lda_composed.mat")
        with mfa_open(log_path, "a") as log_file:
            macc_list = []
            for x in arguments:
                macc_list.extend(x.macc_paths.values())
            subprocess.call(
                [thirdparty_binary("est-mllt"), new_mat_path] + macc_list,
                stderr=log_file,
                env=os.environ,
            )
            subprocess.call(
                [
                    thirdparty_binary("gmm-transform-means"),
                    new_mat_path,
                    self.model_path,
                    self.model_path,
                ],
                stderr=log_file,
                env=os.environ,
            )

            if os.path.exists(previous_mat_path):
                subprocess.call(
                    [
                        thirdparty_binary("compose-transforms"),
                        new_mat_path,
                        previous_mat_path,
                        composed_path,
                    ],
                    stderr=log_file,
                    env=os.environ,
                )
                os.remove(previous_mat_path)
                os.rename(composed_path, previous_mat_path)
            else:
                os.rename(new_mat_path, previous_mat_path)

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
