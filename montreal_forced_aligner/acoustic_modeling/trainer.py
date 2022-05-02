"""Class definitions for trainable aligners"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from montreal_forced_aligner.abc import ModelExporterMixin, TopLevelMfaWorker
from montreal_forced_aligner.alignment.base import CorpusAligner
from montreal_forced_aligner.db import Dictionary
from montreal_forced_aligner.exceptions import ConfigError, KaldiProcessingError
from montreal_forced_aligner.helper import load_configuration, parse_old_features
from montreal_forced_aligner.models import AcousticModel, DictionaryModel
from montreal_forced_aligner.utils import log_kaldi_errors

if TYPE_CHECKING:
    from argparse import Namespace

    from montreal_forced_aligner.abc import MetaDict
    from montreal_forced_aligner.acoustic_modeling.base import AcousticModelTrainingMixin

__all__ = ["TrainableAligner"]


class TrainableAligner(CorpusAligner, TopLevelMfaWorker, ModelExporterMixin):
    """
    Train acoustic model

    Parameters
    ----------
    training_configuration : list[tuple[str, dict[str, Any]]]
        Training identifiers and parameters for training blocks
    phone_set_type: str
        Type of phone set to use for acoustic modeling

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

    def __init__(
        self,
        training_configuration: List[Tuple[str, Dict[str, Any]]] = None,
        phone_set_type: str = None,
        **kwargs,
    ):
        self.param_dict = {
            k: v
            for k, v in kwargs.items()
            if not k.endswith("_directory")
            and not k.endswith("_path")
            and k not in ["clean", "num_jobs", "speaker_characters"]
        }
        self.final_identifier = None
        self.current_subset: int = 0
        self.current_aligner: Optional[AcousticModelTrainingMixin] = None
        self.current_trainer: Optional[AcousticModelTrainingMixin] = None
        self.current_acoustic_model: Optional[AcousticModel] = None
        super().__init__(**kwargs)
        if phone_set_type and phone_set_type != "UNKNOWN":
            self.dictionary_model = DictionaryModel(
                self.dictionary_model.path, phone_set_type=phone_set_type
            )
        self.phone_set_type = self.dictionary_model.phone_set_type
        os.makedirs(self.output_directory, exist_ok=True)
        self.training_configs: Dict[str, AcousticModelTrainingMixin] = {}
        if training_configuration is None:
            training_configuration = TrainableAligner.default_training_configurations()
        for k, v in training_configuration:
            self.add_config(k, v)

    @classmethod
    def default_training_configurations(cls) -> List[Tuple[str, Dict[str, Any]]]:
        """Default MFA training configuration"""
        training_params = []
        training_params.append(("monophone", {"subset": 10000, "boost_silence": 1.25}))
        training_params.append(
            (
                "triphone",
                {
                    "subset": 20000,
                    "boost_silence": 1.25,
                    "num_leaves": 2000,
                    "max_gaussians": 10000,
                },
            )
        )
        training_params.append(
            ("lda", {"subset": 20000, "num_leaves": 2500, "max_gaussians": 15000})
        )
        training_params.append(
            ("sat", {"subset": 20000, "num_leaves": 2500, "max_gaussians": 15000})
        )
        training_params.append(
            ("sat", {"subset": 50000, "num_leaves": 4200, "max_gaussians": 40000})
        )
        training_params.append(("pronunciation_probabilities", {"subset": 50000}))
        training_params.append(
            ("sat", {"subset": 150000, "num_leaves": 5000, "max_gaussians": 100000})
        )
        training_params.append(
            ("pronunciation_probabilities", {"subset": 150000, "optional": True})
        )
        training_params.append(
            (
                "sat",
                {
                    "subset": 0,
                    "num_leaves": 7000,
                    "optional": True,
                    "max_gaussians": 150000,
                    "num_iterations": 20,
                    "quick": True,
                },
            )
        )
        return training_params

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
        if config_path:
            data = load_configuration(config_path)
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
                    if v is None and k in cls.nullable_fields:
                        v = []
                    global_params[k] = v
            if training_params:
                use_default = False
        if use_default:  # default training configuration
            training_params = TrainableAligner.default_training_configurations()
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
            self.write_training_information()
            for config in self.training_configs.values():
                if isinstance(config, str):
                    continue
                config.non_silence_phones = self.non_silence_phones
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                import logging

                logger = logging.getLogger(self.identifier)
                log_kaldi_errors(e.error_logs, logger)
                e.update_log_file(logger)
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
        from montreal_forced_aligner.acoustic_modeling.lda import LdaTrainer
        from montreal_forced_aligner.acoustic_modeling.monophone import MonophoneTrainer
        from montreal_forced_aligner.acoustic_modeling.pronunciation_probabilities import (  # noqa
            PronunciationProbabilityTrainer,
        )
        from montreal_forced_aligner.acoustic_modeling.sat import SatTrainer
        from montreal_forced_aligner.acoustic_modeling.triphone import TriphoneTrainer

        p = {}
        p.update(self.param_dict)
        p.update(params)
        identifier = train_type
        index = 2
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
        elif train_type == "pronunciation_probabilities":
            p = {
                k: v
                for k, v in p.items()
                if k in PronunciationProbabilityTrainer.get_configuration_parameters()
            }
            previous_trainer = self.training_configs[list(self.training_configs.keys())[-1]]
            config = PronunciationProbabilityTrainer(
                identifier=identifier, previous_trainer=previous_trainer, worker=self, **p
            )
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
        if "pronunciation_probabilities" in self.training_configs:
            export_directory = os.path.dirname(output_model_path)
            os.makedirs(export_directory, exist_ok=True)
            silence_probs = self.training_configs[
                "pronunciation_probabilities"
            ].silence_probabilities
            with self.session() as session:
                for (d_id,) in session.query(Dictionary.id):
                    base_name = self.dictionary_base_names[d_id]
                    output_dictionary_path = os.path.join(export_directory, base_name)
                    self.export_lexicon(
                        d_id,
                        output_dictionary_path,
                        probability=True,
                        silence_probabilities=silence_probs,
                    )
        self.training_configs[self.final_identifier].export_model(output_model_path)
        self.log_info(f"Saved model to {output_model_path}")

    @property
    def tree_path(self) -> str:
        """Tree path of the final model"""
        return self.training_configs[self.final_identifier].tree_path

    def train(self) -> None:
        """
        Run through the training configurations to produce a final acoustic model
        """
        self.setup()
        previous = None
        begin = time.time()
        for trainer in self.training_configs.values():
            if self.current_subset is None and trainer.optional:
                self.log_info(
                    "Exiting training early to save time as the corpus is below the subset size for later training stages"
                )
                break
            if trainer.subset < self.num_utterances:
                self.current_subset = trainer.subset
            else:
                self.current_subset = None
                trainer.subset = 0
            if previous is not None:
                self.current_aligner = previous
                os.makedirs(self.working_directory, exist_ok=True)
                self.current_acoustic_model = AcousticModel(
                    previous.exported_model_path, self.working_directory
                )
                self.align()
            if trainer.identifier.startswith("pronunciation_probabilities"):
                trainer.train_pronunciation_probabilities()
            else:
                trainer.train()
            previous = trainer
            self.final_identifier = trainer.identifier
        self.log_info(f"Completed training in {time.time()-begin} seconds!")

        self.current_subset = None
        self.current_aligner = previous
        os.makedirs(self.working_log_directory, exist_ok=True)
        self.current_acoustic_model = AcousticModel(
            previous.exported_model_path, self.working_directory
        )

    def export_files(self, output_directory: str, output_format: Optional[str] = None) -> None:
        """
        Export a TextGrid file for every sound file in the dataset

        Parameters
        ----------
        output_directory: str
            Directory to save to
        """
        self.align()
        super(TrainableAligner, self).export_files(output_directory, output_format)

    @property
    def num_current_utterances(self) -> int:
        """Number of utterances in the current subset"""
        if self.current_subset and self.current_subset < self.num_utterances:
            return self.current_subset
        return self.num_utterances

    @property
    def align_options(self) -> MetaDict:
        """Alignment options"""
        if self.current_aligner is not None:
            return self.current_aligner.align_options
        return super().align_options

    def align(self) -> None:
        """
        Multiprocessing function that aligns based on the current model.

        See Also
        --------
        :class:`~montreal_forced_aligner.alignment.multiprocessing.AlignFunction`
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
            self.log_debug(f"Skipping {self.current_aligner.identifier} alignments")
            return
        try:
            self.current_acoustic_model.export_model(self.working_directory)
            self.compile_train_graphs()
            self.align_utterances()
            if self.current_acoustic_model.meta["features"]["uses_speaker_adaptation"]:

                arguments = self.calc_fmllr_arguments()
                missing_transforms = False
                for arg in arguments:
                    for path in arg.trans_paths.values():
                        if not os.path.exists(path):
                            missing_transforms = True
                if missing_transforms:
                    self.speaker_independent = True
                    assert self.alignment_model_path.endswith(".alimdl")
                    self.calc_fmllr()
                self.speaker_independent = False
                assert self.alignment_model_path.endswith(".mdl")
                self.align_utterances()
            if self.current_subset:
                self.log_debug(
                    f"Analyzing alignment diagnostics for {self.current_aligner.identifier} on {self.current_subset} utterances"
                )
            else:
                self.log_debug(
                    f"Analyzing alignment diagnostics for {self.current_aligner.identifier} on the full corpus"
                )
            self.compile_information()
            with open(done_path, "w"):
                pass
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                import logging

                logger = logging.getLogger(self.identifier)
                log_kaldi_errors(e.error_logs, logger)
                e.update_log_file(logger)
            raise

    @property
    def alignment_model_path(self) -> str:
        """Current alignment model path"""
        path = os.path.join(self.working_directory, "final.alimdl")
        if os.path.exists(path) and self.speaker_independent:
            return path
        return self.model_path

    @property
    def model_path(self) -> str:
        """Current model path"""
        if self.current_trainer is not None:
            return self.current_trainer.model_path
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
        return os.path.join(self.output_directory, f"{self.current_aligner.identifier}_ali")

    @property
    def working_log_directory(self) -> Optional[str]:
        """Current log directory"""
        return os.path.join(self.working_directory, "log")
