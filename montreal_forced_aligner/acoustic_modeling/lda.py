"""Class definitions for LDA trainer"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import TYPE_CHECKING, NamedTuple

from montreal_forced_aligner.acoustic_modeling.triphone import TriphoneTrainer
from montreal_forced_aligner.utils import parse_logs, run_mp, run_non_mp, thirdparty_binary

if TYPE_CHECKING:
    from montreal_forced_aligner.abc import MetaDict


__all__ = ["LdaTrainer"]


class LdaAccStatsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.acoustic_modeling.lda.lda_acc_stats_func`"""

    log_path: str
    dictionaries: list[str]
    feature_strings: dict[str, str]
    ali_paths: dict[str, str]
    model_path: str
    lda_options: MetaDict
    acc_paths: dict[str, str]


class CalcLdaMlltArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.acoustic_modeling.lda.calc_lda_mllt_func`"""

    log_path: str
    """Log file to save stderr"""
    dictionaries: list[str]
    feature_strings: dict[str, str]
    ali_paths: dict[str, str]
    model_path: str
    lda_options: MetaDict
    macc_paths: dict[str, str]


def lda_acc_stats_func(
    log_path: str,
    dictionaries: list[str],
    feature_strings: dict[str, str],
    ali_paths: dict[str, str],
    model_path: str,
    lda_options: MetaDict,
    acc_paths: dict[str, str],
) -> None:
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
    log_path: str
        Path to save log output
    dictionaries: list[str]
        List of dictionary names
    feature_strings: dict[str, str]
        Dictionary of feature strings per dictionary name
    ali_paths: dict[str, str]
        Dictionary of alignment archives per dictionary name
    model_path: str
        Path to the acoustic model file
    lda_options: dict[str, Any]
        Options for LDA
    acc_paths: dict[str, str]
        Dictionary of accumulated stats files per dictionary name
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            ali_path = ali_paths[dict_name]
            feature_string = feature_strings[dict_name]
            acc_path = acc_paths[dict_name]
            ali_to_post_proc = subprocess.Popen(
                [thirdparty_binary("ali-to-post"), f"ark:{ali_path}", "ark:-"],
                stderr=log_file,
                stdout=subprocess.PIPE,
                env=os.environ,
            )
            weight_silence_post_proc = subprocess.Popen(
                [
                    thirdparty_binary("weight-silence-post"),
                    f"{lda_options['boost_silence']}",
                    lda_options["silence_csl"],
                    model_path,
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
                    f"--rand-prune={lda_options['random_prune']}",
                    model_path,
                    feature_string,
                    "ark,s,cs:-",
                    acc_path,
                ],
                stdin=weight_silence_post_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            acc_lda_post_proc.communicate()


def calc_lda_mllt_func(
    log_path: str,
    dictionaries: list[str],
    feature_strings: dict[str, str],
    ali_paths: dict[str, str],
    model_path: str,
    lda_options: MetaDict,
    macc_paths: dict[str, str],
) -> None:
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
    log_path: str
        Path to save log output
    dictionaries: list[str]
        List of dictionary names
    feature_strings: dict[str, str]
        Dictionary of feature strings per dictionary name
    ali_paths: dict[str, str]
        Dictionary of alignment archives per dictionary name
    model_path: str
        Path to the acoustic model file
    lda_options: dict[str, Any]
        Options for LDA
    macc_paths: dict[str, str]
        Dictionary of accumulated stats files per dictionary name
    """
    # Estimating MLLT
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            ali_path = ali_paths[dict_name]
            feature_string = feature_strings[dict_name]
            macc_path = macc_paths[dict_name]
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
                    lda_options["silence_csl"],
                    model_path,
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
                    f"--rand-prune={lda_options['random_prune']}",
                    model_path,
                    feature_string,
                    "ark,s,cs:-",
                    macc_path,
                ],
                stdin=weight_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            acc_proc.communicate()


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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.subset = subset
        self.num_leaves = num_leaves
        self.max_gaussians = max_gaussians
        self.lda_dimension = lda_dimension
        self.random_prune = random_prune
        self.uses_splices = uses_splices
        self.splice_left_context = splice_left_context
        self.splice_right_context = splice_right_context

    def lda_acc_stats_arguments(self) -> list[LdaAccStatsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.acoustic_modeling.lda.lda_acc_stats_func`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.acoustic_modeling.lda.LdaAccStatsArguments`]
            Arguments for processing
        """
        feat_strings = self.worker.construct_feature_proc_strings()
        return [
            LdaAccStatsArguments(
                os.path.join(self.working_log_directory, f"lda_acc_stats.{j.name}.log"),
                j.current_dictionary_names,
                feat_strings[j.name],
                j.construct_path_dictionary(self.previous_aligner.working_directory, "ali", "ark"),
                self.previous_aligner.alignment_model_path,
                self.lda_options,
                j.construct_path_dictionary(self.working_directory, "lda", "acc"),
            )
            for j in self.jobs
        ]

    def calc_lda_mllt_arguments(self) -> list[CalcLdaMlltArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.acoustic_modeling.lda.calc_lda_mllt_func`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.acoustic_modeling.lda.CalcLdaMlltArguments`]
            Arguments for processing
        """
        feat_strings = self.worker.construct_feature_proc_strings()
        return [
            CalcLdaMlltArguments(
                os.path.join(
                    self.working_log_directory, f"lda_mllt.{self.iteration}.{j.name}.log"
                ),
                j.current_dictionary_names,
                feat_strings[j.name],
                j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                self.model_path,
                self.lda_options,
                j.construct_path_dictionary(self.working_directory, "lda", "macc"),
            )
            for j in self.jobs
        ]

    @property
    def train_type(self) -> str:
        """Training identifier"""
        return "lda"

    @property
    def lda_options(self) -> MetaDict:
        """Options for computing LDA"""
        return {
            "lda_dimension": self.lda_dimension,
            "boost_silence": self.boost_silence,
            "random_prune": self.random_prune,
            "silence_csl": self.silence_csl,
        }

    def compute_calculated_properties(self) -> None:
        """Generate realignment iterations, MLLT estimation iterations, and initial gaussians based on configuration"""
        super().compute_calculated_properties()
        self.mllt_iterations = []
        max_mllt_iter = int(self.num_iterations / 2) - 1
        for i in range(1, max_mllt_iter):
            if i < max_mllt_iter / 2 and i % 2 == 0:
                self.mllt_iterations.append(i)
        self.mllt_iterations.append(max_mllt_iter)

    def lda_acc_stats(self) -> None:
        """
        Multiprocessing function that accumulates LDA statistics.

        See Also
        --------
        :func:`~montreal_forced_aligner.acoustic_modeling.lda.lda_acc_stats_func`
            Multiprocessing helper function for each job
        :meth:`.LdaTrainer.lda_acc_stats_arguments`
            Job method for generating arguments for the helper function
        :kaldi_src:`est-lda`
            Relevant Kaldi binary
        :kaldi_steps:`train_lda_mllt`
            Reference Kaldi script

        """
        arguments = self.lda_acc_stats_arguments()

        if self.use_mp:
            run_mp(lda_acc_stats_func, arguments, self.working_log_directory)
        else:
            run_non_mp(lda_acc_stats_func, arguments, self.working_log_directory)

        log_path = os.path.join(self.working_log_directory, "lda_est.log")
        acc_list = []
        for x in arguments:
            acc_list.extend(x.acc_paths.values())
        with open(log_path, "w", encoding="utf8") as log_file:
            est_lda_proc = subprocess.Popen(
                [
                    thirdparty_binary("est-lda"),
                    f"--dim={self.lda_dimension}",
                    os.path.join(self.working_directory, "lda.mat"),
                ]
                + acc_list,
                stderr=log_file,
                env=os.environ,
            )
            est_lda_proc.communicate()
        shutil.copyfile(
            os.path.join(self.working_directory, "lda.mat"),
            os.path.join(self.worker.working_directory, "lda.mat"),
        )

    def _trainer_initialization(self) -> None:
        """Initialize LDA training"""
        self.uses_splices = True
        self.worker.uses_splices = True
        self.lda_acc_stats()
        super()._trainer_initialization()

    def calc_lda_mllt(self) -> None:
        """
        Multiprocessing function that calculates LDA+MLLT transformations.

        See Also
        --------
        :func:`~montreal_forced_aligner.acoustic_modeling.lda.calc_lda_mllt_func`
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
        jobs = self.calc_lda_mllt_arguments()

        if self.use_mp:
            run_mp(calc_lda_mllt_func, jobs, self.working_log_directory)
        else:
            run_non_mp(calc_lda_mllt_func, jobs, self.working_log_directory)

        log_path = os.path.join(
            self.working_log_directory, f"transform_means.{self.iteration}.log"
        )
        previous_mat_path = os.path.join(self.working_directory, "lda.mat")
        new_mat_path = os.path.join(self.working_directory, "lda_new.mat")
        composed_path = os.path.join(self.working_directory, "lda_composed.mat")
        with open(log_path, "a", encoding="utf8") as log_file:
            macc_list = []
            for x in jobs:
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

    def train_iteration(self):
        """
        Run a single LDA training iteration
        """
        if os.path.exists(self.next_model_path):
            return
        if self.iteration in self.realignment_iterations:
            self.align_utterances()
            if self.debug:
                self.compute_alignment_improvement()
        if self.iteration in self.mllt_iterations:
            self.calc_lda_mllt()

        self.acc_stats()
        parse_logs(self.working_log_directory)
        if self.iteration < self.final_gaussian_iteration:
            self.increment_gaussians()
        self.iteration += 1
