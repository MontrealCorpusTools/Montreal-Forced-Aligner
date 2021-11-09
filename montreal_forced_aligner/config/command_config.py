"""Class definitions for configuring commands"""
from __future__ import annotations

import os
from typing import Any

import yaml

__all__ = ["CommandConfig", "load_command_configuration"]


class CommandConfig(object):
    """
    Configuration for running commands

    """

    def __init__(self, data: dict):
        self.data = data

    def __getitem__(self, item: str) -> Any:
        """Get key"""
        if item not in self.data:
            return None
        return self.data[item]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set key"""
        self.data[key] = value

    def update(self, new_data: dict) -> None:
        """Update configuration"""
        self.data.update(new_data)

    def save(self, conf_path: str) -> None:
        """Export to path"""
        with open(conf_path, "w") as f:
            yaml.dump(self.data, f)


def load_command_configuration(conf_path: str, default: dict) -> CommandConfig:
    """
    Load a previous run of MFA in a temporary directory

    Parameters
    ----------
    conf_path: str
        Path to saved configuration
    default: dict
        Extra parameters to set on load

    Returns
    -------
    :class:`~montreal_forced_aligner.config.command_config.CommandConfig`
        Command configuration
    """
    if os.path.exists(conf_path):
        with open(conf_path, "r") as f:
            conf = yaml.load(f, Loader=yaml.SafeLoader)
        config = CommandConfig(conf)
    else:
        config = CommandConfig(default)
    return config
