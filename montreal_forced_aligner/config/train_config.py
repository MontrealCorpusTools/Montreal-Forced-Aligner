"""Class definitions for configuring acoustic model training"""
from __future__ import annotations

import os
from collections import Counter
from typing import Iterator, List, Tuple

import yaml

from ..exceptions import ConfigError
from ..trainers import (
    BaseTrainer,
    IvectorExtractorTrainer,
    LdaTrainer,
    MonophoneTrainer,
    SatTrainer,
    TriphoneTrainer,
)
from .align_config import AlignConfig
from .base_config import BaseConfig
from .dictionary_config import DictionaryConfig
from .feature_config import FeatureConfig

__all__ = [
    "TrainingConfig",
    "train_yaml_to_config",
    "load_basic_train",
    "load_basic_train_ivector",
    "load_test_config",
    "load_sat_adapt",
    "load_no_sat_adapt",
]


class TrainingConfig(BaseConfig):
    """
    Configuration class for storing parameters and trainers for training acoustic models
    """

    def __init__(self, training_configs):
        self.training_configs = training_configs
        counts = Counter([x.train_type for x in self.training_configs])
        self.training_identifiers = []
        curs = {x.train_type: 1 for x in self.training_configs}
        for t in training_configs:
            i = t.train_type
            if counts[t.train_type] != 1:
                i += str(curs[t.train_type])
                curs[t.train_type] += 1
            self.training_identifiers.append(i)

    def update_from_align(self, align_config: AlignConfig) -> None:
        """Update parameters from an AlignConfig"""
        for tc in self.training_configs:
            tc.overwrite = align_config.overwrite
            tc.cleanup_textgrids = align_config.cleanup_textgrids

    def update(self, data: dict) -> None:
        """Update parameters"""
        for k, v in data.items():
            if not hasattr(self, k):
                continue
            setattr(self, k, v)
        for trainer in self.values():
            trainer.update(data)

    def keys(self) -> List:
        """List of training identifiers"""
        return self.training_identifiers

    def values(self) -> List[BaseTrainer]:
        """List of trainers"""
        return self.training_configs

    def items(self) -> Iterator:
        """Iterator over training identifiers and trainers"""
        return zip(self.training_identifiers, self.training_configs)

    def __getitem__(self, item: str) -> BaseTrainer:
        """Get trainer based on identifier"""
        if item not in self.training_identifiers:
            raise KeyError(f"{item} not a valid training identifier")
        return self.training_configs[self.training_identifiers.index(item)]

    @property
    def uses_sat(self) -> bool:
        """Flag for whether a trainer uses speaker adaptation"""
        for k in self.keys():
            if k.startswith("sat"):
                return True
        return False


def train_yaml_to_config(
    path: str, require_mono: bool = True
) -> Tuple[TrainingConfig, AlignConfig, DictionaryConfig]:
    """
    Helper function to load acoustic model training configurations

    Parameters
    ----------
    path: str
        Path to yaml file

    Returns
    -------
    :class:`~montreal_forced_aligner.config.TrainingConfig`
        Training configuration
    :class:`~montreal_forced_aligner.config.AlignConfig`
        Alignment configuration
    :class:`~montreal_forced_aligner.config.DictionaryConfig`
        Dictionary configuration
    """
    dictionary_config = DictionaryConfig()
    with open(path, "r", encoding="utf8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        global_params = {}
        training = []
        training_params = []
        global_feature_params = {}
        for k, v in data.items():
            if k == "training":
                for t in v:
                    for k2, v2 in t.items():
                        feature_config = FeatureConfig()
                        if k2 == "monophone":
                            training.append(MonophoneTrainer(feature_config))
                        elif k2 == "triphone":
                            training.append(TriphoneTrainer(feature_config))
                        elif k2 == "lda":
                            training.append(LdaTrainer(feature_config))
                        elif k2 == "sat":
                            training.append(SatTrainer(feature_config))
                        elif k2 == "ivector":
                            training.append(IvectorExtractorTrainer(feature_config))
                        training_params.append(v2)
            elif k == "features":
                global_feature_params.update(v)
            else:
                global_params[k] = v
        feature_config = FeatureConfig()
        feature_config.update(global_feature_params)
        align_config = AlignConfig(feature_config)
        align_config.update(global_params)
        dictionary_config.update(global_params)
        training_config = None
        if training:
            for i, t in enumerate(training):
                if i == 0 and require_mono and t.train_type not in ["mono", "ivector"]:
                    raise ConfigError("The first round of training must be monophone.")
                t.update(global_params)
                t.update(training_params[i])
                t.feature_config.update(global_feature_params)
            training_config = TrainingConfig(training)
        align_config.feature_config.fmllr = training_config.uses_sat
        if align_config.beam >= align_config.retry_beam:
            raise ConfigError("Retry beam must be greater than beam.")
        return training_config, align_config, dictionary_config


def load_basic_train() -> Tuple[TrainingConfig, AlignConfig, DictionaryConfig]:
    """
    Helper function to load the default parameters

    Returns
    -------
    :class:`~montreal_forced_aligner.config.TrainingConfig`
        Training configuration
    :class:`~montreal_forced_aligner.config.AlignConfig`
        Alignment configuration
    :class:`~montreal_forced_aligner.config.DictionaryConfig`
        Dictionary configuration
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_config, align_config, dictionary_config = train_yaml_to_config(
        os.path.join(base_dir, "basic_train.yaml")
    )
    return training_config, align_config, dictionary_config


def load_sat_adapt() -> Tuple[TrainingConfig, AlignConfig, DictionaryConfig]:
    """
    Helper function to load the default speaker adaptation parameters for adapting an acoustic model to new data

    Returns
    -------
    :class:`~montreal_forced_aligner.config.TrainingConfig`
        Training configuration
    :class:`~montreal_forced_aligner.config.AlignConfig`
        Alignment configuration
    :class:`~montreal_forced_aligner.config.DictionaryConfig`
        Dictionary configuration
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_config, align_config, dictionary_config = train_yaml_to_config(
        os.path.join(base_dir, "adapt_sat.yaml"), require_mono=False
    )
    training_config.training_configs[0].fmllr_iterations = range(
        0, training_config.training_configs[0].num_iterations
    )
    training_config.training_configs[0].realignment_iterations = range(
        0, training_config.training_configs[0].num_iterations
    )
    return training_config, align_config, dictionary_config


def load_no_sat_adapt() -> Tuple[TrainingConfig, AlignConfig, DictionaryConfig]:
    """
    Helper function to load the default parameters for adapting an acoustic model to new data without speaker adaptation

    Returns
    -------
    :class:`~montreal_forced_aligner.config.TrainingConfig`
        Training configuration
    :class:`~montreal_forced_aligner.config.AlignConfig`
        Alignment configuration
    :class:`~montreal_forced_aligner.config.DictionaryConfig`
        Dictionary configuration
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_config, align_config, dictionary_config = train_yaml_to_config(
        os.path.join(base_dir, "adapt_nosat.yaml"), require_mono=False
    )
    training_config.training_configs[0].realignment_iterations = range(
        0, training_config.training_configs[0].num_iterations
    )
    return training_config, align_config, dictionary_config


def load_basic_train_ivector() -> Tuple[TrainingConfig, AlignConfig, DictionaryConfig]:
    """
    Helper function to load the default parameters for training ivector extractors

    Returns
    -------
    :class:`~montreal_forced_aligner.config.TrainingConfig`
        Training configuration
    :class:`~montreal_forced_aligner.config.AlignConfig`
        Alignment configuration
    :class:`~montreal_forced_aligner.config.DictionaryConfig`
        Dictionary configuration
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_config, align_config, dictionary_config = train_yaml_to_config(
        os.path.join(base_dir, "basic_train_ivector.yaml")
    )
    return training_config, align_config, dictionary_config


def load_test_config() -> Tuple[TrainingConfig, AlignConfig, DictionaryConfig]:
    """
    Helper function to load the default parameters for validating corpora

    Returns
    -------
    :class:`~montreal_forced_aligner.config.TrainingConfig`
        Training configuration
    :class:`~montreal_forced_aligner.config.AlignConfig`
        Alignment configuration
    :class:`~montreal_forced_aligner.config.DictionaryConfig`
        Dictionary configuration
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_config, align_config, dictionary_config = train_yaml_to_config(
        os.path.join(base_dir, "test_config.yaml")
    )
    return training_config, align_config, dictionary_config
