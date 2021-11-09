"""Class definitions for configuring speaker classification"""
from __future__ import annotations

import os

import yaml

from .base_config import BaseConfig

__all__ = [
    "SpeakerClassificationConfig",
    "classification_yaml_to_config",
    "load_basic_classification",
]


class SpeakerClassificationConfig(BaseConfig):
    """
    Configuration class to store parameters for speaker classification
    """

    def __init__(self):
        self.use_mp = True
        self.pca_dimension = -1
        self.target_energy = 0.1
        self.cluster_threshold = 0.5
        self.max_speaker_fraction = 1.0
        self.first_pass_max_utterances = 32767
        self.rttm_channel = 0
        self.read_costs = False
        self.overwrite = False


def classification_yaml_to_config(path: str) -> SpeakerClassificationConfig:
    """
    Helper function to load speaker classification configurations

    Parameters
    ----------
    path: str
        Path to yaml file

    Returns
    -------
    :class:`~montreal_forced_aligner.config.speaker_classification_config.SpeakerClassificationConfig`
        Speaker classification configuration
    """
    with open(path, "r", encoding="utf8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        classification_config = SpeakerClassificationConfig()
        if data:
            classification_config.update(data)
        return classification_config


def load_basic_classification() -> SpeakerClassificationConfig:
    """
    Helper function to load the default parameters

    Returns
    -------
    :class:`~montreal_forced_aligner.config.speaker_classification_config.SpeakerClassificationConfig`
        Default speaker classification configuration
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    classification_config = classification_yaml_to_config(
        os.path.join(base_dir, "basic_classification.yaml")
    )
    return classification_config
