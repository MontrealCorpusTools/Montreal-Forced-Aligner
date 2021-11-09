"""Class definitions for configuring G2P generation"""
from __future__ import annotations

import yaml

from .base_config import (
    DEFAULT_CLITIC_MARKERS,
    DEFAULT_COMPOUND_MARKERS,
    DEFAULT_PUNCTUATION,
    BaseConfig,
    ConfigError,
)

__all__ = ["G2PConfig", "g2p_yaml_to_config", "load_basic_g2p_config"]


class G2PConfig(BaseConfig):
    """
    Configuration class for generating pronunciations

    """

    def __init__(self):
        self.punctuation = DEFAULT_PUNCTUATION
        self.clitic_markers = DEFAULT_CLITIC_MARKERS
        self.compound_markers = DEFAULT_COMPOUND_MARKERS
        self.num_pronunciations = 1
        self.use_mp = True

    def update(self, data: dict) -> None:
        """Update configuration"""
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


def g2p_yaml_to_config(path: str) -> G2PConfig:
    """
    Helper function to load G2P configurations

    Parameters
    ----------
    path: str
        Path to yaml file

    Returns
    -------
    :class:`~montreal_forced_aligner.config.g2p_config.G2PConfig`
        G2P configuration
    """
    with open(path, "r", encoding="utf8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        global_params = {}
        for k, v in data.items():
            global_params[k] = v
        g2p_config = G2PConfig()
        g2p_config.update(global_params)
        return g2p_config


def load_basic_g2p_config() -> G2PConfig:
    """
    Helper function to load the default parameters

    Returns
    -------
    :class:`~montreal_forced_aligner.config.g2p_config.G2PConfig`
        Default G2P configuration
    """
    return G2PConfig()
