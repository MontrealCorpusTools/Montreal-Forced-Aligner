"""Class definitions for Monophone trainer"""
from __future__ import annotations

import os
import re
import subprocess
from typing import NamedTuple

from montreal_forced_aligner.acoustic_modeling.base import AcousticModelTrainingMixin
from montreal_forced_aligner.utils import run_mp, run_non_mp, thirdparty_binary


class MonoAlignEqualArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.acoustic_modeling.monophone.mono_align_equal_func`"""

    log_path: str
    dictionaries: list[str]
    feature_strings: dict[str, str]
    fst_scp_paths: dict[str, str]
    ali_ark_paths: dict[str, str]
    acc_paths: dict[str, str]
    model_path: str


def mono_align_equal_func(
    log_path: str,
    dictionaries: list[str],
    feature_strings: dict[str, str],
    fst_scp_paths: dict[str, str],
    ali_ark_paths: dict[str, str],
    acc_paths: dict[str, str],
    model_path: str,
):
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
    log_path: str
        Path to save log output
    dictionaries: list[str]
        List of dictionary names
    feature_strings: dict[str, str]
        Dictionary of feature strings per dictionary name
    fst_scp_paths: dict[str, str]
        Dictionary of utterance FST scp files per dictionary name
    ali_ark_paths: dict[str, str]
        Dictionary of alignment archives per dictionary name
    acc_paths: dict[str, str]
        Dictionary of accumulated stats files per dictionary name
    model_path: str
        Path to the acoustic model file
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            fst_path = fst_scp_paths[dict_name]
            ali_path = ali_ark_paths[dict_name]
            acc_path = acc_paths[dict_name]
            align_proc = subprocess.Popen(
                [
                    thirdparty_binary("align-equal-compiled"),
                    f"scp:{fst_path}",
                    feature_strings[dict_name],
                    f"ark:{ali_path}",
                ],
                stderr=log_file,
                env=os.environ,
            )
            align_proc.communicate()
            stats_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-acc-stats-ali"),
                    "--binary=true",
                    model_path,
                    feature_strings[dict_name],
                    f"ark:{ali_path}",
                    acc_path,
                ],
                stdin=align_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            stats_proc.communicate()


__all__ = ["MonophoneTrainer"]


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
        max_gaussians: int = 1000,
        power: float = 0.25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.subset = subset
        self.initial_gaussians = initial_gaussians
        self.max_gaussians = max_gaussians
        self.power = power

    def mono_align_equal_arguments(self) -> list[MonoAlignEqualArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.acoustic_modeling.monophone.mono_align_equal_func`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.acoustic_modeling.monophone.MonoAlignEqualArguments`]
            Arguments for processing
        """
        feat_strings = self.worker.construct_feature_proc_strings()
        return [
            MonoAlignEqualArguments(
                os.path.join(self.working_log_directory, f"mono_align_equal.{j.name}.log"),
                j.current_dictionary_names,
                feat_strings[j.name],
                j.construct_path_dictionary(self.working_directory, "fsts", "scp"),
                j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                j.construct_path_dictionary(self.working_directory, "0", "acc"),
                self.model_path,
            )
            for j in self.jobs
        ]

    def compute_calculated_properties(self) -> None:
        """Generate realignment iterations and initial gaussians based on configuration"""
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

    def mono_align_equal(self):
        """
        Multiprocessing function that creates equal alignments for base monophone training.

        See Also
        --------
        :func:`~montreal_forced_aligner.acoustic_modeling.monophone.mono_align_equal_func`
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

        arguments = self.mono_align_equal_arguments()

        if self.use_mp:
            run_mp(mono_align_equal_func, arguments, self.working_log_directory)
        else:
            run_non_mp(mono_align_equal_func, arguments, self.working_log_directory)

        log_path = os.path.join(self.working_log_directory, "update.0.log")
        with open(log_path, "w") as log_file:
            acc_files = []
            for x in arguments:
                acc_files.extend(sorted(x.acc_paths.values()))
            sum_proc = subprocess.Popen(
                [thirdparty_binary("gmm-sum-accs"), "-"] + acc_files,
                stderr=log_file,
                stdout=subprocess.PIPE,
                env=os.environ,
            )
            est_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-est"),
                    "--min-gaussian-occupancy=3",
                    f"--mix-up={self.current_gaussians}",
                    f"--power={self.power}",
                    self.model_path,
                    "-",
                    self.next_model_path,
                ],
                stderr=log_file,
                stdin=sum_proc.stdout,
                env=os.environ,
            )
            est_proc.communicate()
        if not self.debug:
            for f in acc_files:
                os.remove(f)

    def _trainer_initialization(self) -> None:
        """Monophone training initialization"""
        self.iteration = 0
        tree_path = os.path.join(self.working_directory, "tree")

        feat_dim = self.worker.get_feat_dim()

        feature_string = self.worker.construct_base_feature_string()
        shared_phones_path = os.path.join(self.worker.phones_dir, "sets.int")
        init_log_path = os.path.join(self.working_log_directory, "init.log")
        temp_feats_path = os.path.join(self.working_directory, "temp_feats")
        with open(init_log_path, "w") as log_file:
            subprocess.call(
                [
                    thirdparty_binary("subset-feats"),
                    "--n=10",
                    feature_string,
                    f"ark:{temp_feats_path}",
                ],
                stderr=log_file,
            )
            subprocess.call(
                [
                    thirdparty_binary("gmm-init-mono"),
                    f"--shared-phones={shared_phones_path}",
                    f"--train-feats=ark:{temp_feats_path}",
                    os.path.join(self.worker.topo_path),
                    str(feat_dim),
                    self.model_path,
                    tree_path,
                ],
                stderr=log_file,
            )
            proc = subprocess.Popen(
                [thirdparty_binary("gmm-info"), "--print-args=false", self.model_path],
                stderr=log_file,
                stdout=subprocess.PIPE,
            )
            stdout, stderr = proc.communicate()
            num = stdout.decode("utf8")
            matches = re.search(r"gaussians (\d+)", num)
            num_gauss = int(matches.groups()[0])
        if os.path.exists(self.model_path):
            os.remove(init_log_path)
        os.remove(temp_feats_path)
        self.initial_gaussians = num_gauss
        self.current_gaussians = num_gauss
        self.compile_train_graphs()
        self.mono_align_equal()
