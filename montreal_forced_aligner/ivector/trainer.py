"""Class definition for TrainableIvectorExtractor"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple

import yaml

from ..abc import MetaDict, ModelExporterMixin, TopLevelMfaWorker
from ..acoustic_modeling.base import AcousticModelTrainingMixin
from ..corpus.features import IvectorConfigMixin
from ..corpus.ivector_corpus import IvectorCorpusMixin
from ..exceptions import ConfigError, KaldiProcessingError
from ..models import IvectorExtractorModel
from ..utils import log_kaldi_errors, parse_logs, run_mp, run_non_mp, thirdparty_binary

if TYPE_CHECKING:
    from argparse import Namespace

__all__ = [
    "TrainableIvectorExtractor",
    "DubmTrainer",
    "IvectorTrainer",
    "IvectorModelTrainingMixin",
    "acc_ivector_stats_func",
]


class IvectorModelTrainingMixin(AcousticModelTrainingMixin):
    """
    Abstract mixin for training ivector extractor models

    See Also
    --------
    :class:`~montreal_forced_aligner.acoustic_modeling.base.AcousticModelTrainingMixin`
        For acoustic model training parsing parameters
    """

    @property
    def meta(self) -> MetaDict:
        """Generate metadata for the acoustic model that was trained"""
        from datetime import datetime

        from ..utils import get_mfa_version

        data = {
            "version": get_mfa_version(),
            "architecture": self.architecture,
            "train_date": str(datetime.now()),
            "features": self.feature_options,
        }
        return data

    def export_model(self, output_model_path: str) -> None:
        """
        Output IvectorExtractor model

        Parameters
        ----------
        output_model_path : str
            Path to save ivector extractor model
        """
        directory, filename = os.path.split(output_model_path)
        basename, _ = os.path.splitext(filename)
        ivector_extractor = IvectorExtractorModel.empty(basename, self.working_log_directory)
        ivector_extractor.add_meta_file(self)
        ivector_extractor.add_model(self.working_directory)
        if directory:
            os.makedirs(directory, exist_ok=True)
        basename, _ = os.path.splitext(output_model_path)
        ivector_extractor.dump(basename)


class GmmGselectArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.ivector.trainer.gmm_gselect_func`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ivector_options: MetaDict
    dubm_model: str
    gselect_paths: Dict[str, str]


class AccGlobalStatsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.ivector.trainer.acc_global_stats_func`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ivector_options: MetaDict
    gselect_paths: Dict[str, str]
    acc_paths: Dict[str, str]
    dubm_path: str


class GaussToPostArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.ivector.trainer.gauss_to_post_func`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ivector_options: MetaDict
    post_paths: Dict[str, str]
    dubm_path: str


class AccIvectorStatsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.ivector.trainer.acc_ivector_stats_func`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ivector_options: MetaDict
    ie_path: str
    post_paths: Dict[str, str]
    acc_init_paths: Dict[str, str]


def gmm_gselect_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    dubm_options: MetaDict,
    dubm_path: str,
    gselect_paths: Dict[str, str],
) -> None:
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
    log_path: str
        Path to save log output
    dictionaries: list[str]
        List of dictionary names
    feature_strings: dict[str, str]
        Dictionary of feature strings per dictionary name
    dubm_options: dict[str, Any]
        Options for DUBM training
    dubm_path: str
        Path to the DUBM file
    gselect_paths: dict[str, str]
        Dictionary of gselect archives per dictionary name
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            gselect_path = gselect_paths[dict_name]
            subsample_feats_proc = subprocess.Popen(
                [
                    thirdparty_binary("subsample-feats"),
                    f"--n={dubm_options['subsample']}",
                    feature_string,
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )

            gselect_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-gselect"),
                    f"--n={dubm_options['num_gselect']}",
                    dubm_path,
                    "ark:-",
                    f"ark:{gselect_path}",
                ],
                stdin=subsample_feats_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            gselect_proc.communicate()


def gauss_to_post_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    ivector_options: MetaDict,
    post_paths: Dict[str, str],
    dubm_path: str,
):
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
    log_path: str
        Path to save log output
    dictionaries: list[str]
        List of dictionary names
    feature_strings: dict[str, str]
        Dictionary of feature strings per dictionary name
    ivector_options: dict[str, Any]
        Options for ivector extractor training
    post_paths: dict[str, str]
        Dictionary of posterior archives per dictionary name
    dubm_path: str
        Path to the DUBM file
    """
    modified_posterior_scale = ivector_options["posterior_scale"] * ivector_options["subsample"]
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            post_path = post_paths[dict_name]
            subsample_feats_proc = subprocess.Popen(
                [
                    thirdparty_binary("subsample-feats"),
                    f"--n={ivector_options['subsample']}",
                    feature_string,
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            gmm_global_get_post_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-global-get-post"),
                    f"--n={ivector_options['num_gselect']}",
                    f"--min-post={ivector_options['min_post']}",
                    dubm_path,
                    "ark:-",
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stdin=subsample_feats_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            scale_post_proc = subprocess.Popen(
                [
                    thirdparty_binary("scale-post"),
                    "ark:-",
                    str(modified_posterior_scale),
                    f"ark:{post_path}",
                ],
                stdin=gmm_global_get_post_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            scale_post_proc.communicate()


def acc_global_stats_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    dubm_options: MetaDict,
    gselect_paths: Dict[str, str],
    acc_paths: Dict[str, str],
    dubm_path: str,
) -> None:
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
    log_path: str
        Path to save log output
    dictionaries: list[str]
        List of dictionary names
    feature_strings: dict[str, str]
        Dictionary of feature strings per dictionary name
    dubm_options: dict[str, Any]
        Options for DUBM training
    gselect_paths: dict[str, str]
        Dictionary of gselect archives per dictionary name
    acc_paths: dict[str, str]
        Dictionary of accumulated stats files per dictionary name
    dubm_path: str
        Path to the DUBM file
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            gselect_path = gselect_paths[dict_name]
            acc_path = acc_paths[dict_name]
            subsample_feats_proc = subprocess.Popen(
                [
                    thirdparty_binary("subsample-feats"),
                    f"--n={dubm_options['subsample']}",
                    feature_string,
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            gmm_global_acc_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-global-acc-stats"),
                    f"--gselect=ark:{gselect_path}",
                    dubm_path,
                    "ark:-",
                    acc_path,
                ],
                stderr=log_file,
                stdin=subsample_feats_proc.stdout,
                env=os.environ,
            )
            gmm_global_acc_proc.communicate()


def acc_ivector_stats_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    ivector_options: MetaDict,
    ie_path: str,
    post_paths: Dict[str, str],
    acc_init_paths: Dict[str, str],
) -> None:
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
    log_path: str
        Path to save log output
    dictionaries: list[str]
        List of dictionary names
    feature_strings: dict[str, str]
        Dictionary of feature strings per dictionary name
    ivector_options: dict[str, Any]
        Options for ivector extractor training
    ie_path: str
        Path to the ivector extractor file
    post_paths: dict[str, str]
        Dictionary of posterior archives per dictionary name
    acc_init_paths: dict[str, str]
        Dictionary of accumulated stats files per dictionary name
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            post_path = post_paths[dict_name]
            acc_init_path = acc_init_paths[dict_name]
            subsample_feats_proc = subprocess.Popen(
                [
                    thirdparty_binary("subsample-feats"),
                    f"--n={ivector_options['subsample']}",
                    feature_string,
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            acc_stats_proc = subprocess.Popen(
                [
                    thirdparty_binary("ivector-extractor-acc-stats"),
                    "--num-threads=1",
                    ie_path,
                    "ark:-",
                    f"ark:{post_path}",
                    acc_init_path,
                ],
                stdin=subsample_feats_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            acc_stats_proc.communicate()


class DubmTrainer(IvectorModelTrainingMixin):
    """
    Trainer for diagonal universal background models

    Parameters
    ----------
    num_iterations : int
        Number of training iterations to perform, defaults to 4
    num_gselect: int
        Number of Gaussian-selection indices to use while training
    subsample: int
        Subsample factor for feature frames, defaults to 5
    num_frames:int
        Number of frames to keep in memory for initialization, defaults to 500000
    num_gaussians:int
        Number of gaussians to use for DUBM training, defaults to 256
    num_iterations_init:int
        Number of iteration to use when initializing UBM, defaults to 20
    initial_gaussian_proportion:float
        Proportion of total gaussians to use initially, defaults to 0.5
    min_gaussian_weight: float
        Defaults to 0.0001
    remove_low_count_gaussians: bool
        Flag for removing low count gaussians in the final round of training, defaults to True

    See Also
    --------
    :class:`~montreal_forced_aligner.ivector.trainer.IvectorModelTrainingMixin`
        For base ivector training parameters
    """

    def __init__(
        self,
        num_iterations: int = 4,
        num_gselect: int = 30,
        subsample: int = 5,
        num_frames: int = 500000,
        num_gaussians: int = 256,
        num_iterations_init: int = 20,
        initial_gaussian_proportion: float = 0.5,
        min_gaussian_weight: float = 0.0001,
        remove_low_count_gaussians: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_iterations = num_iterations
        self.subsample = subsample
        self.num_gselect = num_gselect
        self.num_frames = num_frames
        self.num_gaussians = num_gaussians
        self.num_iterations_init = num_iterations_init
        self.initial_gaussian_proportion = initial_gaussian_proportion
        self.min_gaussian_weight = min_gaussian_weight
        self.remove_low_count_gaussians = remove_low_count_gaussians

    def compute_calculated_properties(self) -> None:
        """Not implemented"""
        pass

    @property
    def train_type(self) -> str:
        """Training identifier"""
        return "dubm"

    @property
    def dubm_options(self):
        """Options for DUBM training"""
        return {"subsample": self.subsample, "num_gselect": self.num_gselect}

    def gmm_gselect_arguments(self) -> List[GmmGselectArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.ivector.trainer.gmm_gselect_func`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.ivector.trainer.GmmGselectArguments`]
            Arguments for processing
        """
        feat_strings = self.construct_feature_proc_strings()
        return [
            GmmGselectArguments(
                os.path.join(self.working_log_directory, f"gmm_gselect.{j.name}.log"),
                j.current_dictionary_names,
                feat_strings[j.name],
                self.dubm_options,
                self.model_path,
                j.construct_path_dictionary(self.working_directory, "gselect", "ark"),
            )
            for j in self.jobs
        ]

    def acc_global_stats_arguments(
        self,
    ) -> List[AccGlobalStatsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.ivector.trainer.acc_global_stats_func`


        Returns
        -------
        list[:class:`~montreal_forced_aligner.ivector.trainer.AccGlobalStatsArguments`]
            Arguments for processing
        """
        feat_strings = self.construct_feature_proc_strings()
        return [
            AccGlobalStatsArguments(
                os.path.join(
                    self.working_log_directory,
                    f"acc_global_stats.{self.iteration}.{j.name}.log",
                ),
                j.current_dictionary_names,
                feat_strings[j.name],
                self.dubm_options,
                j.construct_path_dictionary(self.working_directory, "gselect", "ark"),
                j.construct_path_dictionary(
                    self.working_directory, f"global.{self.iteration}", "acc"
                ),
                self.model_path,
            )
            for j in self.jobs
        ]

    def gmm_gselect(self) -> None:
        """
        Multiprocessing function that stores Gaussian selection indices on disk

        See Also
        --------
        :func:`~montreal_forced_aligner.ivector.trainer.gmm_gselect_func`
            Multiprocessing helper function for each job
        :meth:`.DubmTrainer.gmm_gselect_arguments`
            Job method for generating arguments for the helper function
        :kaldi_steps:`train_diag_ubm`
            Reference Kaldi script

        """
        jobs = self.gmm_gselect_arguments()
        if self.use_mp:
            run_mp(gmm_gselect_func, jobs, self.working_log_directory)
        else:
            run_non_mp(gmm_gselect_func, jobs, self.working_log_directory)

    def _trainer_initialization(self, initial_alignment_directory: Optional[str] = None) -> None:
        """DUBM training initialization"""
        # Initialize model from E-M in memory
        log_directory = os.path.join(self.working_directory, "log")
        if initial_alignment_directory and os.path.exists(initial_alignment_directory):
            jobs = self.align_arguments()
            for j in jobs:
                for p in j.ali_paths.values():
                    shutil.copyfile(
                        p.replace(self.working_directory, initial_alignment_directory), p
                    )
            shutil.copyfile(
                os.path.join(initial_alignment_directory, "final.mdl"),
                os.path.join(self.working_directory, "final.mdl"),
            )
        num_gauss_init = int(self.initial_gaussian_proportion * int(self.num_gaussians))
        log_path = os.path.join(log_directory, "gmm_init.log")
        all_feats_path = os.path.join(self.corpus_output_directory, "feats.scp")
        feature_string = self.construct_base_feature_string(all_feats=True)
        with open(all_feats_path, "w") as outf:
            for i in self.jobs:
                feat_paths = i.construct_path_dictionary(self.data_directory, "feats", "scp")
                for p in feat_paths.values():
                    with open(p) as inf:
                        for line in inf:
                            outf.write(line)
        self.iteration = 1
        with open(log_path, "w") as log_file:
            gmm_init_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-global-init-from-feats"),
                    f"--num-threads={self.worker.num_jobs}",
                    f"--num-frames={self.num_frames}",
                    f"--num_gauss={self.num_gaussians}",
                    f"--num_gauss_init={num_gauss_init}",
                    f"--num_iters={self.num_iterations_init}",
                    feature_string,
                    self.model_path,
                ],
                stderr=log_file,
            )
            gmm_init_proc.communicate()
        # Store Gaussian selection indices on disk
        self.gmm_gselect()
        parse_logs(log_directory)

    def acc_global_stats(self) -> None:
        """
        Multiprocessing function that accumulates global GMM stats

        See Also
        --------
        :func:`~montreal_forced_aligner.ivector.trainer.acc_global_stats_func`
            Multiprocessing helper function for each job
        :meth:`.DubmTrainer.acc_global_stats_arguments`
            Job method for generating arguments for the helper function
        :kaldi_src:`gmm-global-sum-accs`
            Relevant Kaldi binary
        :kaldi_steps:`train_diag_ubm`
            Reference Kaldi script

        """
        jobs = self.acc_global_stats_arguments()
        if self.use_mp:
            run_mp(acc_global_stats_func, jobs, self.working_log_directory)
        else:
            run_non_mp(acc_global_stats_func, jobs, self.working_log_directory)

        # Don't remove low-count Gaussians till the last tier,
        # or gselect info won't be valid anymore
        if self.iteration < self.num_iterations:
            opt = "--remove-low-count-gaussians=false"
        else:
            opt = f"--remove-low-count-gaussians={self.remove_low_count_gaussians}"
        log_path = os.path.join(self.working_log_directory, f"update.{self.iteration}.log")
        with open(log_path, "w") as log_file:
            acc_files = []
            for j in jobs:
                acc_files.extend(j.acc_paths.values())
            sum_proc = subprocess.Popen(
                [thirdparty_binary("gmm-global-sum-accs"), "-"] + acc_files,
                stderr=log_file,
                stdout=subprocess.PIPE,
                env=os.environ,
            )
            gmm_global_est_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-global-est"),
                    opt,
                    f"--min-gaussian-weight={self.min_gaussian_weight}",
                    self.model_path,
                    "-",
                    self.next_model_path,
                ],
                stderr=log_file,
                stdin=sum_proc.stdout,
                env=os.environ,
            )
            gmm_global_est_proc.communicate()
            # Clean up
            if not self.debug:
                for p in acc_files:
                    os.remove(p)

    @property
    def exported_model_path(self) -> str:
        """Temporary model path to save intermediate model"""
        return os.path.join(self.working_log_directory, "dubm_model.zip")

    def train_iteration(self) -> None:
        """
        Run an iteration of UBM training
        """
        # Accumulate stats
        self.acc_global_stats()
        self.iteration += 1

    def finalize_training(self) -> None:
        """Finalize DUBM training"""
        final_dubm_path = os.path.join(self.working_directory, "final.dubm")
        shutil.copy(
            os.path.join(self.working_directory, f"{self.num_iterations+1}.dubm"),
            final_dubm_path,
        )
        self.export_model(self.exported_model_path)
        self.training_complete = True

    @property
    def model_path(self) -> str:
        """Current iteration's DUBM model path"""
        if self.training_complete:
            return os.path.join(self.working_directory, "final.dubm")
        return os.path.join(self.working_directory, f"{self.iteration}.dubm")

    @property
    def next_model_path(self) -> str:
        """Next iteration's DUBM model path"""
        if self.training_complete:
            return os.path.join(self.working_directory, "final.dubm")
        return os.path.join(self.working_directory, f"{self.iteration + 1}.dubm")


class IvectorTrainer(IvectorModelTrainingMixin, IvectorConfigMixin):
    """
    Trainer for a block of ivector extractor training

    Parameters
    ----------
    num_iterations: int
        Number of iterations, defaults to 10
    subsample: int
        Subsample factor for feature frames, defaults to 5
    gaussian_min_count: int

    See Also
    --------
    :class:`~montreal_forced_aligner.ivector.trainer.IvectorModelTrainingMixin`
        For base parameters for ivector training
    :class:`~montreal_forced_aligner.corpus.features.IvectorConfigMixin`
        For parameters for ivector feature generation

    """

    def __init__(
        self, num_iterations: int = 10, subsample: int = 5, gaussian_min_count: int = 100, **kwargs
    ):
        super().__init__(**kwargs)
        self.subsample = subsample
        self.num_iterations = num_iterations
        self.gaussian_min_count = gaussian_min_count

    def compute_calculated_properties(self) -> None:
        """Not implemented"""
        pass

    @property
    def exported_model_path(self) -> str:
        """Temporary directory path that trainer will save ivector extractor model"""
        return os.path.join(self.working_log_directory, "ivector_model.zip")

    def acc_ivector_stats_arguments(self) -> List[AccIvectorStatsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.ivector.trainer.acc_ivector_stats_func`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.ivector.trainer.AccIvectorStatsArguments`]
            Arguments for processing
        """
        feat_strings = self.construct_feature_proc_strings()
        arguments = [
            AccIvectorStatsArguments(
                os.path.join(self.working_log_directory, f"ivector_acc.{j.name}.log"),
                j.current_dictionary_names,
                feat_strings[j.name],
                self.ivector_options,
                self.ie_path,
                j.construct_path_dictionary(self.working_directory, "post", "ark"),
                j.construct_path_dictionary(self.working_directory, "ivector", "acc"),
            )
            for j in self.jobs
        ]

        return arguments

    def _trainer_initialization(self) -> None:
        """Ivector extractor training initialization"""
        self.iteration = 1
        self.training_complete = False
        # Initialize job_name-vector extractor
        log_directory = os.path.join(self.working_directory, "log")
        log_path = os.path.join(log_directory, "init.log")
        diag_ubm_path = os.path.join(self.working_directory, "final.dubm")

        full_ubm_path = os.path.join(self.working_directory, "final.ubm")
        with open(log_path, "w") as log_file:
            subprocess.call(
                [thirdparty_binary("gmm-global-to-fgmm"), diag_ubm_path, full_ubm_path],
                stderr=log_file,
            )
            subprocess.call(
                [
                    thirdparty_binary("ivector-extractor-init"),
                    f"--ivector-dim={self.ivector_dimension}",
                    "--use-weights=false",
                    full_ubm_path,
                    self.ie_path,
                ],
                stderr=log_file,
            )

        # Do Gaussian selection and posterior extraction
        self.gauss_to_post()
        parse_logs(log_directory)

    def gauss_to_post_arguments(self) -> List[GaussToPostArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.ivector.trainer.gauss_to_post_func`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.ivector.trainer.GaussToPostArguments`]
            Arguments for processing
        """
        feat_strings = self.construct_feature_proc_strings()
        return [
            GaussToPostArguments(
                os.path.join(self.working_log_directory, f"gauss_to_post.{j.name}.log"),
                j.current_dictionary_names,
                feat_strings[j.name],
                self.ivector_options,
                j.construct_path_dictionary(self.working_directory, "post", "ark"),
                self.dubm_path,
            )
            for j in self.jobs
        ]

    def gauss_to_post(self) -> None:
        """
        Multiprocessing function that does Gaussian selection and posterior extraction

        See Also
        --------
        :func:`~montreal_forced_aligner.ivector.trainer.gauss_to_post_func`
            Multiprocessing helper function for each job
        :meth:`.IvectorTrainer.gauss_to_post_arguments`
            Job method for generating arguments for the helper function
        :kaldi_steps_sid:`train_ivector_extractor`
            Reference Kaldi script
        """
        jobs = self.gauss_to_post_arguments()
        if self.use_mp:
            run_mp(gauss_to_post_func, jobs, self.working_log_directory)
        else:
            run_non_mp(gauss_to_post_func, jobs, self.working_log_directory)

    @property
    def train_type(self) -> str:
        """Training identifier"""
        return "ivector"

    @property
    def ivector_options(self) -> MetaDict:
        """Options for ivector training and extracting"""
        options = super().ivector_options
        options["subsample"] = self.subsample
        return options

    @property
    def meta(self) -> MetaDict:
        """Metadata information for ivector extractor models"""
        from ..utils import get_mfa_version

        return {
            "version": get_mfa_version(),
            "ivector_dimension": self.ivector_dimension,
            "num_gselect": self.num_gselect,
            "min_post": self.min_post,
            "posterior_scale": self.posterior_scale,
            "features": self.feature_options,
        }

    @property
    def ie_path(self) -> str:
        """Current ivector extractor model path"""
        if self.training_complete:
            return os.path.join(self.working_directory, "final.ie")
        return os.path.join(self.working_directory, f"{self.iteration}.ie")

    @property
    def next_ie_path(self) -> str:
        """Next iteration's ivector extractor model path"""
        if self.training_complete:
            return os.path.join(self.working_directory, "final.ie")
        return os.path.join(self.working_directory, f"{self.iteration + 1}.ie")

    @property
    def dubm_path(self) -> str:
        """DUBM model path"""
        return os.path.join(self.working_directory, "final.dubm")

    def acc_ivector_stats(self) -> None:
        """
        Multiprocessing function that accumulates ivector extraction stats.

        See Also
        --------
        :func:`~montreal_forced_aligner.ivector.trainer.acc_ivector_stats_func`
            Multiprocessing helper function for each job
        :meth:`.IvectorTrainer.acc_ivector_stats_arguments`
            Job method for generating arguments for the helper function
        :kaldi_src:`ivector-extractor-sum-accs`
            Relevant Kaldi binary
        :kaldi_src:`ivector-extractor-est`
            Relevant Kaldi binary
        :kaldi_steps_sid:`train_ivector_extractor`
            Reference Kaldi script
        """

        jobs = self.acc_ivector_stats_arguments()
        if self.use_mp:
            run_mp(acc_ivector_stats_func, jobs, self.working_log_directory)
        else:
            run_non_mp(acc_ivector_stats_func, jobs, self.working_log_directory)

        log_path = os.path.join(self.working_log_directory, f"sum_acc.{self.iteration}.log")
        acc_path = os.path.join(self.working_directory, f"acc.{self.iteration}")
        with open(log_path, "w", encoding="utf8") as log_file:
            accinits = []
            for j in jobs:
                accinits.extend(j.acc_init_paths.values())
            sum_accs_proc = subprocess.Popen(
                [thirdparty_binary("ivector-extractor-sum-accs"), "--parallel=true"]
                + accinits
                + [acc_path],
                stderr=log_file,
                env=os.environ,
            )

            sum_accs_proc.communicate()
        # clean up
        for p in accinits:
            os.remove(p)
        # Est extractor
        log_path = os.path.join(self.working_log_directory, f"update.{self.iteration}.log")
        with open(log_path, "w") as log_file:
            extractor_est_proc = subprocess.Popen(
                [
                    thirdparty_binary("ivector-extractor-est"),
                    f"--num-threads={len(self.jobs)}",
                    f"--gaussian-min-count={self.gaussian_min_count}",
                    self.ie_path,
                    os.path.join(self.working_directory, f"acc.{self.iteration}"),
                    self.next_ie_path,
                ],
                stderr=log_file,
                env=os.environ,
            )
            extractor_est_proc.communicate()

    def train_iteration(self):
        """
        Run an iteration of training
        """
        if os.path.exists(self.next_ie_path):
            self.iteration += 1
            return
        # Accumulate stats and sum
        self.acc_ivector_stats()

        self.iteration += 1

    def finalize_training(self):
        """
        Finalize ivector extractor training
        """
        # Rename to final
        shutil.copy(
            os.path.join(self.working_directory, f"{self.num_iterations}.ie"),
            os.path.join(self.working_directory, "final.ie"),
        )
        self.training_complete = True


class TrainableIvectorExtractor(IvectorCorpusMixin, TopLevelMfaWorker, ModelExporterMixin):
    """
    Trainer for ivector extractor models

    Parameters
    ----------
    training_configuration: list[tuple[str, dict[str, Any]]]
        Training configurations to use, defaults to a round of dubm training followed by ivector training

    See Also
    --------
    :class:`~montreal_forced_aligner.corpus.ivector_corpus.IvectorCorpusMixin`
        For parameters to parse corpora using ivector features
    :class:`~montreal_forced_aligner.abc.TopLevelMfaWorker`
        For top-level parameters
    :class:`~montreal_forced_aligner.abc.ModelExporterMixin`
        For model export parameters
    """

    def __init__(self, training_configuration: List[Tuple[str, Dict[str, Any]]] = None, **kwargs):
        self.param_dict = {
            k: v
            for k, v in kwargs.items()
            if not k.endswith("_directory")
            and not k.endswith("_path")
            and k not in ["clean", "num_jobs", "speaker_characters"]
        }
        self.final_identifier = None
        super().__init__(**kwargs)
        os.makedirs(self.output_directory, exist_ok=True)
        self.training_configs: Dict[str, AcousticModelTrainingMixin] = {}
        self.current_model = None
        if training_configuration is None:
            training_configuration = [("dubm", {}), ("ivector", {})]
        for k, v in training_configuration:
            self.add_config(k, v)

    def setup(self) -> None:
        """Setup ivector extractor training"""
        if self.initialized:
            return
        self.check_previous_run()
        try:
            self.load_corpus()
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger)
            raise
        self.initialized = True

    def add_config(self, train_type: str, params: MetaDict) -> None:
        """
        Add a trainer to the pipeline

        Parameters
        ----------
        train_type: str
            Type of trainer to add, one of "dubm" or "ivector"
        params: dict[str, Any]
            Parameters to initialize trainer

        Raises
        ------
        ConfigError
            If an invalid ``train_type`` is specified
        """
        p = {}
        p.update(self.param_dict)
        p.update(params)
        identifier = train_type
        index = 1
        while identifier in self.training_configs:
            identifier = f"{train_type}_{index}"
            index += 1
        self.final_identifier = identifier
        if train_type == "dubm":
            config = DubmTrainer(identifier=identifier, worker=self, **p)
        elif train_type == "ivector":
            config = IvectorTrainer(identifier=identifier, worker=self, **p)
        else:
            raise ConfigError(f"Invalid training type '{train_type}' in config file")

        self.training_configs[identifier] = config

    @property
    def workflow_identifier(self) -> str:
        """Ivector training identifier"""
        return "train_ivector"

    @property
    def meta(self) -> MetaDict:
        """Metadata about the final round of training"""
        return self.training_configs[self.final_identifier].meta

    def train(self) -> None:
        """
        Run through the training configurations to produce a final ivector extractor model
        """
        begin = time.time()
        self.setup()
        previous = None
        for trainer in self.training_configs.values():
            self.current_subset = trainer.subset
            if previous is not None:
                self.current_model = IvectorExtractorModel(previous.exported_model_path)
                os.makedirs(trainer.working_directory, exist_ok=True)
                self.current_model.export_model(trainer.working_directory)
            trainer.train()
            previous = trainer
        self.logger.info(f"Completed training in {time.time()-begin} seconds!")

    def export_model(self, output_model_path: str) -> None:
        """
        Export an ivector extractor model to the specified path

        Parameters
        ----------
        output_model_path : str
            Path to save ivector extractor model
        """
        self.training_configs[self.final_identifier].export_model(output_model_path)
        self.logger.info(f"Saved model to {output_model_path}")

    @classmethod
    def parse_parameters(
        cls,
        config_path: Optional[str] = None,
        args: Optional[Namespace] = None,
        unknown_args: Optional[List[str]] = None,
    ) -> MetaDict:
        """
        Parse configuration parameters from a config file and command line arguments

        Parameters
        ----------
        config_path: str, optional
            Path to yaml configuration file
        args: :class:`~argparse.Namespace`, optional
            Arguments parsed by argparse
        unknown_args: list[str], optional
            List of unknown arguments from argparse

        Returns
        -------
        dict[str, Any]
            Dictionary of specified configuration parameters
        """
        global_params = {}
        training_params = []
        use_default = True
        if config_path is not None:
            with open(config_path, "r", encoding="utf8") as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)
                training_params = []
                for k, v in data.items():
                    if k == "training":
                        for t in v:
                            for k2, v2 in t.items():
                                if "features" in v2:
                                    global_params.update(v2["features"])
                                    del v2["features"]
                                training_params.append((k2, v2))
                    elif k == "features":
                        if "type" in v:
                            v["feature_type"] = v["type"]
                            del v["type"]
                        global_params.update(v)
                    else:
                        if v is None and k in {
                            "punctuation",
                            "compound_markers",
                            "clitic_markers",
                        }:
                            v = []
                        global_params[k] = v
                if training_params:
                    use_default = False
        if use_default:  # default training configuration
            training_params.append(("dubm", {}))
            # training_params.append(("ubm", {}))
            training_params.append(("ivector", {}))
        if training_params:
            if training_params[0][0] != "dubm":
                raise ConfigError("The first round of training must be dubm.")
        global_params["training_configuration"] = training_params
        global_params.update(cls.parse_args(args, unknown_args))
        return global_params
