"""Class definitions for trainable aligners"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any, Optional

import yaml

from montreal_forced_aligner.abc import ModelExporterMixin, TopLevelMfaWorker
from montreal_forced_aligner.acoustic_modeling.base import AcousticModelTrainingMixin
from montreal_forced_aligner.acoustic_modeling.lda import LdaTrainer
from montreal_forced_aligner.acoustic_modeling.monophone import MonophoneTrainer
from montreal_forced_aligner.acoustic_modeling.sat import SatTrainer
from montreal_forced_aligner.acoustic_modeling.triphone import TriphoneTrainer
from montreal_forced_aligner.alignment.base import CorpusAligner
from montreal_forced_aligner.exceptions import ConfigError, KaldiProcessingError
from montreal_forced_aligner.helper import parse_old_features
from montreal_forced_aligner.models import AcousticModel
from montreal_forced_aligner.utils import log_kaldi_errors

if TYPE_CHECKING:
    from argparse import Namespace

    from montreal_forced_aligner.abc import MetaDict

__all__ = ["TrainableAligner"]


class TrainableAligner(CorpusAligner, TopLevelMfaWorker, ModelExporterMixin):
    """
    Train acoustic model

    Parameters
    ----------
    training_configuration : list[tuple[str, dict[str, Any]]]
        Training identifiers and parameters for training blocks

    See Also
    --------
    :class:`~montreal_forced_aligner.alignment.base.CorpusAligner`
        For dictionary and corpus parsing parameters and alignment parameters
    :class:`~montreal_forced_aligner.abc.TopLevelMfaWorker`
        For top-level parameters
    :class:`~montreal_forced_aligner.abc.ModelExporterMixin`
        For model export parameters

    Attributes
    ----------
    param_dict: dict[str, Any]
        Parameters to pass to training blocks
    final_identifier: str
        Identifier of the final training block
    current_subset: int
        Current training block's subset
    current_acoustic_model: :class:`~montreal_forced_aligner.models.AcousticModel`
        Acoustic model to use in aligning, based on previous training block
    training_configs: dict[str, :class:`~montreal_forced_aligner.acoustic_modeling.base.AcousticModelTrainingMixin`]
        Training blocks
    """

    def __init__(self, training_configuration: list[tuple[str, dict[str, Any]]] = None, **kwargs):
        self.param_dict = {
            k: v
            for k, v in kwargs.items()
            if not k.endswith("_directory")
            and not k.endswith("_path")
            and k not in ["clean", "num_jobs", "speaker_characters"]
        }
        self.final_identifier = None
        self.current_subset: int = 0
        self.current_aligner = None
        self.current_trainer = None
        self.current_acoustic_model: Optional[AcousticModel] = None
        super().__init__(**kwargs)
        os.makedirs(self.output_directory, exist_ok=True)
        self.training_configs: dict[str, AcousticModelTrainingMixin] = {}
        if training_configuration is None:
            training_configuration = [
                ("monophone", {}),
                ("triphone", {}),
                ("lda", {}),
                ("sat", {}),
                ("sat", {"subset": 0, "num_leaves": 4200, "max_gaussians": 40000}),
            ]
        for k, v in training_configuration:
            self.add_config(k, v)

    @classmethod
    def parse_parameters(
        cls,
        config_path: Optional[str] = None,
        args: Optional[Namespace] = None,
        unknown_args: Optional[list[str]] = None,
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
        if config_path:
            with open(config_path, "r", encoding="utf8") as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)
                training_params = []
                for k, v in data.items():
                    if k == "training":
                        for t in v:
                            for k2, v2 in t.items():
                                if "features" in v2:
                                    global_params.update(parse_old_features(v2["features"]))
                                    del v2["features"]
                                training_params.append((k2, v2))
                    elif k == "features":
                        global_params.update(parse_old_features(v))
                    else:
                        global_params[k] = v
                if not training_params:
                    raise ConfigError(f"No 'training' block found in {config_path}")
        else:  # default training configuration
            training_params.append(("monophone", {}))
            training_params.append(("triphone", {}))
            training_params.append(("lda", {}))
            training_params.append(("sat", {}))
            training_params.append(
                ("sat", {"subset": 0, "num_leaves": 4200, "max_gaussians": 40000})
            )
        if training_params:
            if training_params[0][0] != "monophone":
                raise ConfigError("The first round of training must be monophone.")
        global_params["training_configuration"] = training_params
        global_params.update(cls.parse_args(args, unknown_args))
        return global_params

    def setup(self) -> None:
        """Setup for acoustic model training"""
        if self.initialized:
            return
        self.check_previous_run()
        try:
            self.load_corpus()
            for config in self.training_configs.values():
                config.non_silence_phones = self.non_silence_phones
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger)
            raise
        self.initialized = True

    @property
    def workflow_identifier(self) -> str:
        """Acoustic model training identifier"""
        return "train_acoustic_model"

    @property
    def configuration(self) -> MetaDict:
        """Configuration for the worker"""
        config = super().configuration
        config.update(
            {
                "dictionary_path": self.dictionary_model.path,
                "corpus_directory": self.corpus_directory,
            }
        )
        return config

    @property
    def meta(self) -> MetaDict:
        """Metadata about the final round of training"""
        return self.training_configs[self.final_identifier].meta

    def add_config(self, train_type: str, params: MetaDict) -> None:
        """
        Add a trainer to the pipeline

        Parameters
        ----------
        train_type: str
            Type of trainer to add, one of ``monophone``, ``triphone``, ``lda`` or ``sat``
        params: dict[str, Any]
            Parameters to initialize trainer

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.ConfigError`
            If an invalid train_type is specified
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
        if train_type == "monophone":
            p = {
                k: v for k, v in p.items() if k in MonophoneTrainer.get_configuration_parameters()
            }
            config = MonophoneTrainer(identifier=identifier, worker=self, **p)
        elif train_type == "triphone":
            p = {k: v for k, v in p.items() if k in TriphoneTrainer.get_configuration_parameters()}
            config = TriphoneTrainer(identifier=identifier, worker=self, **p)
        elif train_type == "lda":
            p = {k: v for k, v in p.items() if k in LdaTrainer.get_configuration_parameters()}
            config = LdaTrainer(identifier=identifier, worker=self, **p)
        elif train_type == "sat":
            p = {k: v for k, v in p.items() if k in SatTrainer.get_configuration_parameters()}
            config = SatTrainer(identifier=identifier, worker=self, **p)
        else:
            raise ConfigError(f"Invalid training type '{train_type}' in config file")

        self.training_configs[identifier] = config

    def export_model(self, output_model_path: str) -> None:
        """
        Export an acoustic model to the specified path

        Parameters
        ----------
        output_model_path : str
            Path to save acoustic model
        """
        self.training_configs[self.final_identifier].export_model(output_model_path)
        self.logger.info(f"Saved model to {output_model_path}")

    @property
    def backup_output_directory(self) -> Optional[str]:
        """Backup directory if overwriting files is not allowed"""
        if self.overwrite:
            return None
        return os.path.join(self.working_directory, "textgrids")

    @property
    def tree_path(self) -> str:
        """Tree path of the final model"""
        return self.training_configs[self.final_identifier].tree_path

    def train(self, generate_final_alignments: bool = True) -> None:
        """
        Run through the training configurations to produce a final acoustic model

        Parameters
        ----------
        generate_final_alignments: bool
            Flag for whether final alignments should be generated at the end of training, defaults to True
        """
        self.setup()
        previous = None
        begin = time.time()
        for trainer in self.training_configs.values():
            self.current_subset = trainer.subset
            if previous is not None:
                self.current_aligner = previous.identifier
                os.makedirs(self.working_directory, exist_ok=True)
                self.current_acoustic_model = AcousticModel(
                    previous.exported_model_path, self.working_directory
                )
                self.align()
            trainer.train()
            previous = trainer
        self.logger.info(f"Completed training in {time.time()-begin} seconds!")

        if generate_final_alignments:
            self.current_subset = None
            self.current_aligner = previous.identifier
            os.makedirs(self.working_log_directory, exist_ok=True)
            self.current_acoustic_model = AcousticModel(
                previous.exported_model_path, self.working_directory
            )
            self.align()

    def align(self) -> None:
        """
        Multiprocessing function that aligns based on the current model.

        See Also
        --------
        :func:`~montreal_forced_aligner.alignment.multiprocessing.align_func`
            Multiprocessing helper function for each job
        :meth:`.AlignMixin.align_arguments`
            Job method for generating arguments for the helper function
        :kaldi_steps:`align_si`
            Reference Kaldi script
        :kaldi_steps:`align_fmllr`
            Reference Kaldi script
        """
        done_path = os.path.join(self.working_directory, "done")
        if os.path.exists(done_path):
            self.logger.debug(f"Skipping {self.current_aligner} alignments")
            return
        try:
            self.current_acoustic_model.export_model(self.working_directory)
            self.compile_train_graphs()
            self.align_utterances()
            if self.current_subset:
                self.logger.debug(
                    f"Analyzing alignment diagnostics for {self.current_aligner} on {self.current_subset} utterances"
                )
            else:
                self.logger.debug(
                    f"Analyzing alignment diagnostics for {self.current_aligner} on the full corpus"
                )
            self.compile_information()
            with open(done_path, "w"):
                pass
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger)
            raise

    @property
    def alignment_model_path(self) -> str:
        """Current alignment model path"""
        path = os.path.join(self.working_directory, "final.alimdl")
        if os.path.exists(path):
            return path
        return self.model_path

    @property
    def model_path(self) -> str:
        """Current model path"""
        return os.path.join(self.working_directory, "final.mdl")

    @property
    def data_directory(self) -> str:
        """Current data directory based on the trainer's subset"""
        return self.subset_directory(self.current_subset)

    @property
    def working_directory(self) -> Optional[str]:
        """Working directory"""
        if self.current_trainer is not None:
            return self.current_trainer.working_directory
        if self.current_aligner is None:
            return None
        return os.path.join(self.output_directory, f"{self.current_aligner}_ali")

    @property
    def working_log_directory(self) -> Optional[str]:
        """Current log directory"""
        return os.path.join(self.working_directory, "log")
