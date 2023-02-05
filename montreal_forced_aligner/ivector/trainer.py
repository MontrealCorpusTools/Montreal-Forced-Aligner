"""Class definition for TrainableIvectorExtractor"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
import typing
from typing import Any, Dict, List, Optional, Tuple

import tqdm

from montreal_forced_aligner.abc import MetaDict, ModelExporterMixin, TopLevelMfaWorker
from montreal_forced_aligner.acoustic_modeling.base import AcousticModelTrainingMixin
from montreal_forced_aligner.config import GLOBAL_CONFIG, PLDA_DIMENSION
from montreal_forced_aligner.corpus.features import IvectorConfigMixin
from montreal_forced_aligner.corpus.ivector_corpus import IvectorCorpusMixin
from montreal_forced_aligner.data import WorkflowType
from montreal_forced_aligner.db import CorpusWorkflow
from montreal_forced_aligner.exceptions import ConfigError, KaldiProcessingError
from montreal_forced_aligner.helper import load_configuration, mfa_open
from montreal_forced_aligner.ivector.multiprocessing import (
    AccGlobalStatsArguments,
    AccGlobalStatsFunction,
    AccIvectorStatsArguments,
    AccIvectorStatsFunction,
    GaussToPostArguments,
    GaussToPostFunction,
    GmmGselectArguments,
    GmmGselectFunction,
)
from montreal_forced_aligner.models import IvectorExtractorModel
from montreal_forced_aligner.utils import (
    log_kaldi_errors,
    parse_logs,
    run_kaldi_function,
    thirdparty_binary,
)

__all__ = [
    "TrainableIvectorExtractor",
    "DubmTrainer",
    "IvectorTrainer",
    "IvectorModelTrainingMixin",
]

logger = logging.getLogger("mfa")


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

    def compute_calculated_properties(self) -> None:
        """Not implemented"""
        pass

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
        self.use_alignment_features = False

    @property
    def train_type(self) -> str:
        """Training identifier"""
        return "dubm"

    @property
    def dubm_options(self) -> MetaDict:
        """Options for DUBM training"""
        return {"subsample": self.subsample, "num_gselect": self.num_gselect}

    def gmm_gselect_arguments(self) -> List[GmmGselectArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.ivector.trainer.GmmGselectFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.ivector.trainer.GmmGselectArguments`]
            Arguments for processing
        """
        arguments = []
        for j in self.jobs:
            arguments.append(
                GmmGselectArguments(
                    j.id,
                    getattr(self, "db_string", ""),
                    os.path.join(self.working_log_directory, f"gmm_gselect.{j.id}.log"),
                    self.feature_options,
                    self.dubm_options,
                    self.model_path,
                    j.construct_path(self.working_directory, "gselect", "ark"),
                )
            )
        return arguments

    def acc_global_stats_arguments(
        self,
    ) -> List[AccGlobalStatsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.ivector.trainer.AccGlobalStatsFunction`


        Returns
        -------
        list[:class:`~montreal_forced_aligner.ivector.trainer.AccGlobalStatsArguments`]
            Arguments for processing
        """
        arguments = []
        for j in self.jobs:
            arguments.append(
                AccGlobalStatsArguments(
                    j.id,
                    getattr(self, "db_string", ""),
                    os.path.join(
                        self.working_log_directory,
                        f"acc_global_stats.{self.iteration}.{j.id}.log",
                    ),
                    self.feature_options,
                    self.dubm_options,
                    j.construct_path(self.working_directory, "gselect", "ark"),
                    j.construct_path(self.working_directory, f"global.{self.iteration}", "acc"),
                    self.model_path,
                )
            )
        return arguments

    def gmm_gselect(self) -> None:
        """
        Multiprocessing function that stores Gaussian selection indices on disk

        See Also
        --------
        :func:`~montreal_forced_aligner.ivector.trainer.GmmGselectFunction`
            Multiprocessing helper function for each job
        :meth:`.DubmTrainer.gmm_gselect_arguments`
            Job method for generating arguments for the helper function
        :kaldi_steps:`train_diag_ubm`
            Reference Kaldi script

        """
        begin = time.time()
        logger.info("Selecting gaussians...")
        arguments = self.gmm_gselect_arguments()
        with tqdm.tqdm(
            total=int(self.num_current_utterances / 10), disable=GLOBAL_CONFIG.quiet
        ) as pbar:
            for _ in run_kaldi_function(GmmGselectFunction, arguments, pbar.update):
                pass

        logger.debug(f"Gaussian selection took {time.time() - begin:.3f} seconds")

    def _trainer_initialization(self, initial_alignment_directory: Optional[str] = None) -> None:
        """DUBM training initialization"""
        log_path = os.path.join(self.working_log_directory, "gmm_init.log")
        with self.session() as session, mfa_open(log_path, "w") as log_file:
            alignment_workflow: CorpusWorkflow = (
                session.query(CorpusWorkflow)
                .filter(CorpusWorkflow.workflow_type == WorkflowType.alignment)
                .first()
            )
            num_gauss_init = int(self.initial_gaussian_proportion * int(self.num_gaussians))
            self.iteration = 1
            if self.use_alignment_features and alignment_workflow is not None:
                model_path = os.path.join(alignment_workflow.working_directory, "final.mdl")
                occs_path = os.path.join(alignment_workflow.working_directory, "final.occs")
                gmm_init_proc = subprocess.Popen(
                    [
                        thirdparty_binary("init-ubm"),
                        "--fullcov=false",
                        "--intermediate-num-gauss=2000",
                        f"--num-frames={self.num_frames}",
                        f"--ubm-num-gauss={self.num_gaussians}",
                        model_path,
                        occs_path,
                        self.model_path,
                    ],
                    stderr=log_file,
                )
                gmm_init_proc.communicate()
            else:
                job = self.jobs[0]
                feature_string = job.construct_online_feature_proc_string()
                feature_string = feature_string.replace(f".{job.id}.scp", ".scp")
                feature_string = feature_string.replace(
                    job.corpus.current_subset_directory, job.corpus.data_directory
                )
                gmm_init_proc = subprocess.Popen(
                    [
                        thirdparty_binary("gmm-global-init-from-feats"),
                        "--verbose=4",
                        f"--num-threads={GLOBAL_CONFIG.num_jobs}",
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
            parse_logs(self.working_log_directory)

    def acc_global_stats(self) -> None:
        """
        Multiprocessing function that accumulates global GMM stats

        See Also
        --------
        :func:`~montreal_forced_aligner.ivector.trainer.AccGlobalStatsFunction`
            Multiprocessing helper function for each job
        :meth:`.DubmTrainer.acc_global_stats_arguments`
            Job method for generating arguments for the helper function
        :kaldi_src:`gmm-global-sum-accs`
            Relevant Kaldi binary
        :kaldi_steps:`train_diag_ubm`
            Reference Kaldi script

        """
        begin = time.time()
        logger.info("Accumulating global stats...")
        arguments = self.acc_global_stats_arguments()

        with tqdm.tqdm(total=self.num_current_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
            for _ in run_kaldi_function(AccGlobalStatsFunction, arguments, pbar.update):
                pass

        logger.debug(f"Accumulating stats took {time.time() - begin:.3f} seconds")

        # Don't remove low-count Gaussians till the last tier,
        # or gselect info won't be valid anymore
        if self.iteration < self.num_iterations:
            opt = "--remove-low-count-gaussians=false"
        else:
            opt = f"--remove-low-count-gaussians={self.remove_low_count_gaussians}"
        log_path = os.path.join(self.working_log_directory, f"update.{self.iteration}.log")
        with mfa_open(log_path, "w") as log_file:
            acc_files = []
            for j in arguments:
                acc_files.append(j.acc_path)
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
            if not GLOBAL_CONFIG.debug:
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
        # Update VAD with dubm likelihoods
        self.export_model(self.exported_model_path)
        wf = self.worker.current_workflow
        with self.session() as session:
            session.query(CorpusWorkflow).filter(CorpusWorkflow.id == wf.id).update({"done": True})
            session.commit()

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

    @property
    def exported_model_path(self) -> str:
        """Temporary directory path that trainer will save ivector extractor model"""
        return os.path.join(self.working_log_directory, "ivector_model.zip")

    def acc_ivector_stats_arguments(self) -> List[AccIvectorStatsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.ivector.trainer.AccIvectorStatsFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.ivector.trainer.AccIvectorStatsArguments`]
            Arguments for processing
        """
        arguments = []
        for j in self.jobs:
            arguments.append(
                AccIvectorStatsArguments(
                    j.id,
                    getattr(self, "db_string", ""),
                    os.path.join(
                        self.working_log_directory, f"ivector_acc.{self.iteration}.{j.id}.log"
                    ),
                    self.feature_options,
                    self.ivector_options,
                    self.ie_path,
                    j.construct_path(self.working_directory, "post", "ark"),
                    j.construct_path(self.working_directory, "ivector", "acc"),
                )
            )
        return arguments

    def _trainer_initialization(self) -> None:
        """Ivector extractor training initialization"""
        self.iteration = 1
        # Initialize job_name-vector extractor
        log_directory = os.path.join(self.working_directory, "log")
        log_path = os.path.join(log_directory, "init.log")
        diag_ubm_path = os.path.join(self.working_directory, "final.dubm")

        full_ubm_path = os.path.join(self.working_directory, "final.ubm")
        if not os.path.exists(self.ie_path):
            with mfa_open(log_path, "w") as log_file:
                subprocess.check_call(
                    [thirdparty_binary("gmm-global-to-fgmm"), diag_ubm_path, full_ubm_path],
                    stderr=log_file,
                )
                subprocess.check_call(
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
        Generate Job arguments for :func:`~montreal_forced_aligner.ivector.trainer.GaussToPostFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.ivector.trainer.GaussToPostArguments`]
            Arguments for processing
        """
        arguments = []
        for j in self.jobs:
            arguments.append(
                GaussToPostArguments(
                    j.id,
                    getattr(self, "db_string", ""),
                    os.path.join(self.working_log_directory, f"gauss_to_post.{j.id}.log"),
                    self.feature_options,
                    self.ivector_options,
                    j.construct_path(self.working_directory, "post", "ark"),
                    self.dubm_path,
                )
            )
        return arguments

    def gauss_to_post(self) -> None:
        """
        Multiprocessing function that does Gaussian selection and posterior extraction

        See Also
        --------
        :func:`~montreal_forced_aligner.ivector.trainer.GaussToPostFunction`
            Multiprocessing helper function for each job
        :meth:`.IvectorTrainer.gauss_to_post_arguments`
            Job method for generating arguments for the helper function
        :kaldi_steps_sid:`train_ivector_extractor`
            Reference Kaldi script
        """
        begin = time.time()
        logger.info("Extracting posteriors...")
        arguments = self.gauss_to_post_arguments()

        with tqdm.tqdm(total=self.num_current_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
            for _ in run_kaldi_function(GaussToPostFunction, arguments, pbar.update):
                pass

        logger.debug(f"Extracting posteriors took {time.time() - begin:.3f} seconds")

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
        :func:`~montreal_forced_aligner.ivector.trainer.AccIvectorStatsFunction`
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

        begin = time.time()
        logger.info("Accumulating ivector stats...")
        arguments = self.acc_ivector_stats_arguments()

        with tqdm.tqdm(total=self.worker.num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
            for _ in run_kaldi_function(AccIvectorStatsFunction, arguments, pbar.update):
                pass

        logger.debug(f"Accumulating stats took {time.time() - begin:.3f} seconds")

        log_path = os.path.join(self.working_log_directory, f"sum_acc.{self.iteration}.log")
        acc_path = os.path.join(self.working_directory, f"acc.{self.iteration}")
        with mfa_open(log_path, "w") as log_file:
            accinits = []
            for j in arguments:
                accinits.append(j.acc_path)
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
            if os.path.exists(p):
                os.remove(p)
        # Est extractor
        log_path = os.path.join(self.working_log_directory, f"update.{self.iteration}.log")
        with mfa_open(log_path, "w") as log_file:
            extractor_est_proc = subprocess.Popen(
                [
                    thirdparty_binary("ivector-extractor-est"),
                    f"--num-threads={len(self.jobs)}",
                    f"--gaussian-min-count={self.gaussian_min_count}",
                    self.ie_path,
                    os.path.join(self.working_directory, f"acc.{self.iteration}"),
                    self.next_ie_path,
                ],
                stderr=subprocess.PIPE,
                env=os.environ,
            )
            iteration_improvement = None
            explained_variance = None
            for line in extractor_est_proc.stderr:
                line = line.decode("utf8")
                log_file.write(line)
                log_file.flush()
                m = re.match(
                    r"LOG.*Overall objective-function improvement per frame was (?P<improvement>[0-9.]+)",
                    line,
                )
                if m:
                    iteration_improvement = float(m.group("improvement"))
                m = re.match(
                    r"LOG.*variance explained by the iVectors is (?P<variance>[0-9.]+)\.", line
                )
                if m:
                    explained_variance = float(m.group("variance"))
            extractor_est_proc.wait()
            logger.debug(
                f"For iteration {self.iteration}, objective-function improvement was {iteration_improvement} per frame and variance explained by ivectors was {explained_variance}."
            )

    def train_iteration(self) -> None:
        """
        Run an iteration of training
        """
        if not os.path.exists(self.next_ie_path):
            # Accumulate stats and sum
            self.acc_ivector_stats()
        self.iteration += 1

    def finalize_training(self) -> None:
        """
        Finalize ivector extractor training
        """
        # Rename to final
        shutil.copy(
            os.path.join(self.working_directory, f"{self.num_iterations}.ie"),
            os.path.join(self.working_directory, "final.ie"),
        )
        self.export_model(self.exported_model_path)
        wf = self.worker.current_workflow
        with self.session() as session:
            session.query(CorpusWorkflow).filter(CorpusWorkflow.id == wf.id).update({"done": True})
            session.commit()


class PldaTrainer(IvectorTrainer):
    """
    Trainer for a PLDA models

    """

    worker: TrainableIvectorExtractor

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _trainer_initialization(self) -> None:
        """No initialization"""
        pass

    def compute_lda(self):

        lda_path = os.path.join(self.working_directory, "ivector_lda.mat")
        log_path = os.path.join(self.working_log_directory, "lda.log")
        utt2spk_path = os.path.join(self.corpus_output_directory, "utt2spk.scp")
        with tqdm.tqdm(
            total=self.worker.num_utterances, disable=GLOBAL_CONFIG.quiet
        ) as pbar, mfa_open(log_path, "w") as log_file:
            normalize_proc = subprocess.Popen(
                [
                    thirdparty_binary("ivector-normalize-length"),
                    f"scp:{self.worker.utterance_ivector_path}",
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            lda_compute_proc = subprocess.Popen(
                [
                    thirdparty_binary("ivector-compute-lda"),
                    f"--dim={PLDA_DIMENSION}",
                    "--total-covariance-factor=0.1",
                    "ark:-",
                    f"ark:{utt2spk_path}",
                    lda_path,
                ],
                stdin=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            for line in normalize_proc.stdout:
                lda_compute_proc.stdin.write(line)
                lda_compute_proc.stdin.flush()
                pbar.update(1)

            lda_compute_proc.stdin.close()
            lda_compute_proc.wait()
        assert os.path.exists(lda_path)

    def train(self):
        """Train PLDA"""
        self.compute_lda()
        self.worker.compute_plda()
        self.worker.compute_speaker_ivectors()
        os.rename(
            os.path.join(self.working_directory, "current_speaker_ivectors.ark"),
            os.path.join(self.working_directory, "speaker_ivectors.ark"),
        )
        os.rename(
            os.path.join(self.working_directory, "current_num_utts.ark"),
            os.path.join(self.working_directory, "num_utts.ark"),
        )


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
            and k not in ["speaker_characters"]
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
        self.uses_voiced = True

    def setup(self) -> None:
        """Setup ivector extractor training"""
        TopLevelMfaWorker.setup(self)
        if self.initialized:
            return
        try:
            super().load_corpus()
            with self.session() as session:
                workflows: typing.Dict[str, CorpusWorkflow] = {
                    x.name: x for x in session.query(CorpusWorkflow)
                }
                for i, (identifier, config) in enumerate(self.training_configs.items()):
                    if isinstance(config, str):
                        continue
                    if identifier not in workflows:
                        self.create_new_current_workflow(
                            WorkflowType.acoustic_training, name=identifier
                        )
                    else:
                        wf = workflows[identifier]
                        if wf.dirty and not wf.done:
                            shutil.rmtree(wf.working_directory, ignore_errors=True)
                        if i == 0:
                            wf.current = True
                session.commit()
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs)
                e.update_log_file()
            raise
        self.initialized = True

    def add_config(self, train_type: str, params: MetaDict) -> None:
        """
        Add a trainer to the pipeline

        Parameters
        ----------
        train_type: str
            Type of trainer to add, one of "dubm", "ivector", or "plda"
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
        elif train_type == "plda":
            config = PldaTrainer(identifier=identifier, worker=self, **p)
        else:
            raise ConfigError(f"Invalid training type '{train_type}' in config file")

        self.training_configs[identifier] = config

    @property
    def meta(self) -> MetaDict:
        """Metadata about the final round of training"""
        return self.training_configs[self.final_identifier].meta

    @property
    def model_path(self) -> str:
        """Current model path"""
        return self.training_configs[self.current_workflow.name].model_path

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
                os.makedirs(trainer.working_log_directory, exist_ok=True)
                self.current_model.export_model(trainer.working_directory)
            self.set_current_workflow(trainer.identifier)
            trainer.train()
            previous = trainer
        logger.info(f"Completed training in {time.time()-begin} seconds!")

    def export_model(self, output_model_path: str) -> None:
        """
        Export an ivector extractor model to the specified path

        Parameters
        ----------
        output_model_path : str
            Path to save ivector extractor model
        """
        self.training_configs[self.final_identifier].export_model(output_model_path)

        logger.info(f"Saved model to {output_model_path}")

    @classmethod
    def parse_parameters(
        cls,
        config_path: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
        unknown_args: Optional[typing.Iterable[str]] = None,
    ) -> MetaDict:
        """
        Parse configuration parameters from a config file and command line arguments

        Parameters
        ----------
        config_path: str, optional
            Path to yaml configuration file
        args: dict[str, Any]
            Parsed arguments
        unknown_args: list[str]
            Optional list of arguments that were not parsed

        Returns
        -------
        dict[str, Any]
            Dictionary of specified configuration parameters
        """
        global_params = {}
        training_params = []
        use_default = True
        if config_path is not None:
            data = load_configuration(config_path)
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
                    if v is None and k in cls.nullable_fields:
                        v = []
                    global_params[k] = v
                if training_params:
                    use_default = False
        if use_default:  # default training configuration
            training_params.append(("dubm", {}))
            training_params.append(("ivector", {}))
            training_params.append(("plda", {}))
        if training_params:
            if training_params[0][0] != "dubm":
                raise ConfigError("The first round of training must be dubm.")
        global_params["training_configuration"] = training_params
        global_params.update(cls.parse_args(args, unknown_args))
        return global_params
