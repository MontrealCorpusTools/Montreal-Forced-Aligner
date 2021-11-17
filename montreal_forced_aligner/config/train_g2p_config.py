"""Class definitions for configuring G2P model training"""
from __future__ import annotations

from typing import Tuple

import yaml

from .base_config import BaseConfig
from .dictionary_config import DictionaryConfig

__all__ = ["TrainG2PConfig", "train_g2p_yaml_to_config", "load_basic_train_g2p_config"]


class TrainG2PConfig(BaseConfig):
    """
    Configuration class for training G2P models
    """

    def __init__(self):
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


def train_g2p_yaml_to_config(path: str) -> Tuple[TrainG2PConfig, DictionaryConfig]:
    """
    Helper function to load G2P training configurations

    Parameters
    ----------
    path: str
        Path to yaml file

    Returns
    -------
    :class:`~montreal_forced_aligner.config.TrainG2PConfig`
        G2P training configuration
    :class:`~montreal_forced_aligner.config.DictionaryConfig`
        Dictionary configuration
    """
    dictionary_config = DictionaryConfig()
    with open(path, "r", encoding="utf8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        global_params = {}
        for k, v in data.items():
            global_params[k] = v
        g2p_config = TrainG2PConfig()
        g2p_config.update(global_params)
        dictionary_config.update(global_params)
    return g2p_config, dictionary_config


def load_basic_train_g2p_config() -> Tuple[TrainG2PConfig, DictionaryConfig]:
    """
    Helper function to load the default parameters

    Returns
    -------
    :class:`~montreal_forced_aligner.config.TrainG2PConfig`
        Default G2P training configuration
    :class:`~montreal_forced_aligner.config.DictionaryConfig`
        Dictionary configuration
    """
    return TrainG2PConfig(), DictionaryConfig()
