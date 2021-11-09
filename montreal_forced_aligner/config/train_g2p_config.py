"""Class definitions for configuring G2P model training"""
from __future__ import annotations

import yaml

from .base_config import (
    DEFAULT_CLITIC_MARKERS,
    DEFAULT_COMPOUND_MARKERS,
    DEFAULT_PUNCTUATION,
    BaseConfig,
    ConfigError,
)

__all__ = ["TrainG2PConfig", "train_g2p_yaml_to_config", "load_basic_train_g2p_config"]


class TrainG2PConfig(BaseConfig):
    """
    Configuration class for training G2P models
    """

    def __init__(self):
        self.punctuation = DEFAULT_PUNCTUATION
        self.clitic_markers = DEFAULT_CLITIC_MARKERS
        self.compound_markers = DEFAULT_COMPOUND_MARKERS
        self.num_pronunciations = 1
        self.order = 7
        self.random_starts = 25
        self.seed = 1917
        self.delta = 1 / 1024
        self.lr = 1.0
        self.batch_size = 200
        self.max_iterations = 10
        self.smoothing_method = "kneser_ney"
        self.pruning_method = "relative_entropy"
        self.model_size = 1000000
        self.use_mp = True

    def update(self, data: dict) -> None:
        """Update configuration parameters"""
        for k, v in data.items():
            if k in ["punctuation", "clitic_markers", "compound_markers"]:
                if not v:
                    continue
                if "-" in v:
                    v = "-" + v.replace("-", "")
                if "]" in v and r"\]" not in v:
                    v = v.replace("]", r"\]")
            elif not hasattr(self, k):
                raise ConfigError("No field found for key {}".format(k))
            setattr(self, k, v)


def train_g2p_yaml_to_config(path: str) -> TrainG2PConfig:
    """
    Helper function to load G2P training configurations

    Parameters
    ----------
    path: str
        Path to yaml file

    Returns
    -------
    :class:`~montreal_forced_aligner.config.train_g2p_config.TrainG2PConfig`
        G2P training configuration
    """
    with open(path, "r", encoding="utf8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        global_params = {}
        for k, v in data.items():
            global_params[k] = v
        g2p_config = TrainG2PConfig()
        g2p_config.update(global_params)
        return g2p_config


def load_basic_train_g2p_config() -> TrainG2PConfig:
    """
    Helper function to load the default parameters

    Returns
    -------
    :class:`~montreal_forced_aligner.config.train_g2p_config.TrainG2PConfig`
        Default G2P training configuration
    """
    return TrainG2PConfig()
