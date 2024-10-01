"""Class definition for TrainableIvectorExtractor"""
from __future__ import annotations

import logging
import os
import random
import shutil
import time
import typing
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _kalpy import ivector
from _kalpy.gmm import AccumDiagGmm, DiagGmm, FullGmm, MleDiagGmmOptions, StringToGmmFlags
from _kalpy.matrix import FloatMatrix, MatrixResizeType
from kalpy.utils import kalpy_logger, read_kaldi_object, write_kaldi_object

from montreal_forced_aligner import config
from montreal_forced_aligner.abc import MetaDict, ModelExporterMixin, TopLevelMfaWorker
from montreal_forced_aligner.acoustic_modeling.base import AcousticModelTrainingMixin
from montreal_forced_aligner.corpus.features import IvectorConfigMixin
from montreal_forced_aligner.corpus.ivector_corpus import IvectorCorpusMixin
from montreal_forced_aligner.data import WorkflowType
from montreal_forced_aligner.db import CorpusWorkflow
from montreal_forced_aligner.exceptions import ConfigError, KaldiProcessingError
from montreal_forced_aligner.helper import load_configuration
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
from montreal_forced_aligner.utils import log_kaldi_errors, run_kaldi_function

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

        meta = {
            "version": get_mfa_version(),
            "architecture": self.architecture,
            "train_date": str(datetime.now()),
            "features": self.feature_options,
        }
        if self.model_version is not None:
            meta["version"] = self.model_version
        return meta

    def refresh_training_graph_compilers(self):
        pass

    def compute_calculated_properties(self) -> None:
        """Not implemented"""
        pass

    def export_model(self, output_model_path: Path) -> None:
        """
        Output IvectorExtractor model

        Parameters
        ----------
        output_model_path : str
            Path to save ivector extractor model
        """
        directory = output_model_path.parent

        ivector_extractor = IvectorExtractorModel.empty(
            output_model_path.stem, self.working_log_directory
        )
        ivector_extractor.add_meta_file(self)
        ivector_extractor.add_model(self.working_directory)
        if directory:
            os.makedirs(directory, exist_ok=True)
        ivector_extractor.dump(output_model_path)


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
        return {
            "subsample": self.subsample,
            "uses_cmvn": self.uses_cmvn,
            "num_gselect": self.num_gselect,
        }

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
                    getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                    self.working_log_directory.joinpath(f"gmm_gselect.{j.id}.log"),
                    self.working_directory,
                    self.model_path,
                    self.dubm_options,
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
                    getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                    os.path.join(
                        self.working_log_directory,
                        f"acc_global_stats.{self.iteration}.{j.id}.log",
                    ),
                    self.working_directory,
                    self.model_path,
                    self.dubm_options,
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
        for _ in run_kaldi_function(
            GmmGselectFunction, arguments, total_count=self.num_current_utterances
        ):
            pass

        logger.debug(f"Gaussian selection took {time.time() - begin:.3f} seconds")

    def _trainer_initialization(self, initial_alignment_directory: Optional[str] = None) -> None:
        """DUBM training initialization"""
        log_path = self.working_log_directory.joinpath("gmm_init.log")
        with kalpy_logger("kalpy.ivector", log_path) as ivector_logger:
            num_gauss_init = int(self.initial_gaussian_proportion * int(self.num_gaussians))
            self.iteration = 1
            job = self.jobs[0]
            options = MleDiagGmmOptions()
            feature_archive = job.construct_feature_archive(self.worker.split_directory)
            feats = FloatMatrix()
            num_read = 0
            dim = 0
            random.seed(config.SEED)
            for _, current_feats in feature_archive:
                for t in range(current_feats.NumRows()):
                    if num_read == 0:
                        dim = current_feats.NumCols()
                        feats.Resize(self.num_frames, current_feats.NumCols())
                    num_read += 1
                    if num_read < self.num_frames:
                        feats.Row(num_read - 1).CopyFromVec(current_feats.Row(t))
                    else:
                        keep_prob = self.num_frames / num_read
                        if random.uniform(0, 1) <= keep_prob:
                            feats.Row(random.randint(0, self.num_frames - 1)).CopyFromVec(
                                current_feats.Row(t)
                            )

            if num_read < self.num_frames:
                ivector_logger.warning(
                    f"Number of frames {num_read} was less than the target number {self.num_frames}, using all frames"
                )
                feats.Resize(num_read, dim, MatrixResizeType.kCopyData)
            else:
                percent = self.num_frames * 100.0 / num_read
                ivector_logger.info(
                    f"Kept {self.num_frames} out of {num_read} input frames = {percent:.2f}%"
                )
            if num_gauss_init <= 0 or num_gauss_init > self.num_gaussians:
                num_gauss_init = self.num_gaussians
            gmm = DiagGmm(num_gauss_init, dim)
            ivector_logger.info(
                f"Initializing GMM means from random frames to {num_gauss_init} Gaussians."
            )
            gmm.init_from_random_frames(feats)
            cur_num_gauss = num_gauss_init
            gauss_inc = int((self.num_gaussians - num_gauss_init) / (self.num_iterations_init / 2))
            for i in range(self.num_iterations_init):
                gmm.train_one_iter(feats, options, i, config.NUM_JOBS)
                next_num_gauss = min(self.num_gaussians, cur_num_gauss, gauss_inc)
                if next_num_gauss > gmm.NumGauss():
                    gmm.Split(next_num_gauss, 0.1)
                    cur_num_gauss = next_num_gauss
            write_kaldi_object(gmm, self.model_path)

            # Store Gaussian selection indices on disk
            self.gmm_gselect()

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
        gmm_accs = AccumDiagGmm()
        model: DiagGmm = read_kaldi_object(DiagGmm, self.model_path)
        gmm_accs.Resize(model, StringToGmmFlags("mvw"))
        for result in run_kaldi_function(
            AccGlobalStatsFunction, arguments, total_count=self.num_current_utterances
        ):
            if isinstance(result, AccumDiagGmm):
                gmm_accs.Add(1.0, result)

        logger.debug(f"Accumulating stats took {time.time() - begin:.3f} seconds")

        remove_low_count_gaussians = self.remove_low_count_gaussians
        if self.iteration < self.num_iterations:
            remove_low_count_gaussians = False
        log_path = self.working_log_directory.joinpath(f"update.{self.iteration}.log")
        with kalpy_logger("kalpy.ivector", log_path):
            model.mle_update(gmm_accs, remove_low_count_gaussians=remove_low_count_gaussians)
            write_kaldi_object(model, self.next_model_path)

    @property
    def exported_model_path(self) -> Path:
        """Temporary model path to save intermediate model"""
        return self.working_log_directory.joinpath("dubm_model.zip")

    def train_iteration(self) -> None:
        """
        Run an iteration of UBM training
        """
        # Accumulate stats
        self.acc_global_stats()
        self.iteration += 1

    def finalize_training(self) -> None:
        """Finalize DUBM training"""
        final_dubm_path = self.working_directory.joinpath("final.dubm")
        shutil.copy(
            self.working_directory.joinpath(f"{self.num_iterations + 1}.dubm"),
            final_dubm_path,
        )
        # Update VAD with dubm likelihoods
        self.export_model(self.exported_model_path)
        wf = self.worker.current_workflow
        with self.session() as session:
            session.query(CorpusWorkflow).filter(CorpusWorkflow.id == wf.id).update({"done": True})
            session.commit()

    @property
    def model_path(self) -> Path:
        """Current iteration's DUBM model path"""
        if self.training_complete:
            return self.working_directory.joinpath("final.dubm")
        return self.working_directory.joinpath(f"{self.iteration}.dubm")

    @property
    def next_model_path(self) -> Path:
        """Next iteration's DUBM model path"""
        if self.training_complete:
            return self.working_directory.joinpath("final.dubm")
        return self.working_directory.joinpath(f"{self.iteration + 1}.dubm")


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
        self.sliding_cmvn = True
        self.num_iterations = num_iterations
        self.gaussian_min_count = gaussian_min_count

    @property
    def exported_model_path(self) -> Path:
        """Temporary directory path that trainer will save ivector extractor model"""
        return self.working_log_directory.joinpath("ivector_model.zip")

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
                    getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                    os.path.join(
                        self.working_log_directory, f"ivector_acc.{self.iteration}.{j.id}.log"
                    ),
                    self.working_directory,
                    self.ie_path,
                    self.ivector_options,
                )
            )
        return arguments

    def _trainer_initialization(self) -> None:
        """Ivector extractor training initialization"""
        self.iteration = 1
        # Initialize job_name-vector extractor
        log_directory = self.working_directory.joinpath("log")
        log_path = os.path.join(log_directory, "init.log")
        diag_ubm_path = self.working_directory.joinpath("final.dubm")

        if not os.path.exists(self.ie_path):
            with kalpy_logger("kalpy.ivector", log_path):
                gmm = read_kaldi_object(DiagGmm, diag_ubm_path)
                fgmm = FullGmm()
                fgmm.CopyFromDiagGmm(gmm)

                options = ivector.IvectorExtractorOptions()
                options.ivector_dim = self.ivector_dimension
                options.use_weights = False
                extractor = ivector.IvectorExtractor(options, fgmm)
                write_kaldi_object(extractor, self.ie_path)
        self.gauss_to_post()

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
                    getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                    self.working_log_directory.joinpath(f"gauss_to_post.{j.id}.log"),
                    self.working_directory,
                    self.dubm_path,
                    self.ivector_options,
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

        for _ in run_kaldi_function(
            GaussToPostFunction, arguments, total_count=self.num_current_utterances
        ):
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
        options["uses_cmvn"] = self.uses_cmvn
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
    def ie_path(self) -> Path:
        """Current ivector extractor model path"""
        if self.training_complete:
            return self.working_directory.joinpath("final.ie")
        return self.working_directory.joinpath(f"{self.iteration}.ie")

    @property
    def next_ie_path(self) -> Path:
        """Next iteration's ivector extractor model path"""
        if self.training_complete:
            return self.working_directory.joinpath("final.ie")
        return self.working_directory.joinpath(f"{self.iteration + 1}.ie")

    @property
    def dubm_path(self) -> Path:
        """DUBM model path"""
        return self.working_directory.joinpath("final.dubm")

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

        model: ivector.IvectorExtractor = read_kaldi_object(ivector.IvectorExtractor, self.ie_path)
        options = ivector.IvectorExtractorStatsOptions()
        ivector_stats = ivector.IvectorExtractorStats(model, options)
        for result in run_kaldi_function(
            AccIvectorStatsFunction, arguments, total_count=self.worker.num_utterances
        ):
            if isinstance(result, ivector.IvectorExtractorStats):
                ivector_stats.Add(result)

        logger.debug(f"Accumulating stats took {time.time() - begin:.3f} seconds")

        begin = time.time()

        ivector_stats.update(
            model,
            gaussian_min_count=self.gaussian_min_count,
            num_threads=config.NUM_JOBS,
        )
        ivector_stats.IvectorVarianceDiagnostic(model)
        write_kaldi_object(model, self.next_ie_path)

        logger.debug(f"Ivector extractor update took {time.time() - begin:.3f} seconds")

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
            self.working_directory.joinpath(f"{self.num_iterations}.ie"),
            self.working_directory.joinpath("final.ie"),
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

    def train(self):
        """Train PLDA"""
        self.worker.compute_plda()
        self.worker.compute_speaker_ivectors()
        os.rename(
            self.working_directory.joinpath("current_speaker_ivectors.ark"),
            self.working_directory.joinpath("speaker_ivectors.ark"),
        )
        os.rename(
            self.working_directory.joinpath("current_num_utts.ark"),
            self.working_directory.joinpath("num_utts.ark"),
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
                for i, (identifier, c) in enumerate(self.training_configs.items()):
                    if isinstance(c, str):
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
        logger.info(f"Completed training in {time.time() - begin} seconds!")

    def export_model(self, output_model_path: Path) -> None:
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
        config_path: Optional[Path] = None,
        args: Optional[Dict[str, Any]] = None,
        unknown_args: Optional[typing.Iterable[str]] = None,
    ) -> MetaDict:
        """
        Parse configuration parameters from a config file and command line arguments

        Parameters
        ----------
        config_path: :class:`~pathlib.Path`, optional
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
