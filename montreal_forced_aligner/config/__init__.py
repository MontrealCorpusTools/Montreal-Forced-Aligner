"""
Configuration classes
=====================


"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from argparse import Namespace

import os

import yaml

from .align_config import AlignConfig, align_yaml_to_config, load_basic_align  # noqa
from .base_config import BaseConfig
from .command_config import CommandConfig, load_command_configuration  # noqa
from .dictionary_config import DictionaryConfig  # noqa
from .feature_config import FeatureConfig  # noqa
from .g2p_config import G2PConfig, g2p_yaml_to_config, load_basic_g2p_config  # noqa
from .segmentation_config import (  # noqa
    SegmentationConfig,
    load_basic_segmentation,
    segmentation_yaml_to_config,
)
from .speaker_classification_config import (  # noqa
    SpeakerClassificationConfig,
    classification_yaml_to_config,
    load_basic_classification,
)
from .train_config import (  # noqa
    TrainingConfig,
    load_basic_train,
    load_basic_train_ivector,
    load_test_config,
    train_yaml_to_config,
)
from .train_g2p_config import (  # noqa
    TrainG2PConfig,
    load_basic_train_g2p_config,
    train_g2p_yaml_to_config,
)
from .train_lm_config import TrainLMConfig, load_basic_train_lm, train_lm_yaml_to_config  # noqa
from .transcribe_config import (  # noqa
    TranscribeConfig,
    load_basic_transcribe,
    transcribe_yaml_to_config,
)

__all__ = [
    "TEMP_DIR",
    "align_config",
    "base_config",
    "command_config",
    "dictionary_config",
    "feature_config",
    "segmentation_config",
    "speaker_classification_config",
    "train_config",
    "train_lm_config",
    "transcribe_config",
    "generate_config_path",
    "generate_command_history_path",
    "load_command_history",
    "update_command_history",
    "update_global_config",
    "load_global_config",
    "USE_COLORS",
    "BLAS_THREADS",
]

BaseConfig.__module__ = "montreal_forced_aligner.config"
AlignConfig.__module__ = "montreal_forced_aligner.config"
CommandConfig.__module__ = "montreal_forced_aligner.config"
FeatureConfig.__module__ = "montreal_forced_aligner.config"
DictionaryConfig.__module__ = "montreal_forced_aligner.config"
SegmentationConfig.__module__ = "montreal_forced_aligner.config"
SpeakerClassificationConfig.__module__ = "montreal_forced_aligner.config"
TrainingConfig.__module__ = "montreal_forced_aligner.config"
TrainLMConfig.__module__ = "montreal_forced_aligner.config"
TrainG2PConfig.__module__ = "montreal_forced_aligner.config"
G2PConfig.__module__ = "montreal_forced_aligner.config"
TranscribeConfig.__module__ = "montreal_forced_aligner.config"


TEMP_DIR = os.path.expanduser("~/Documents/MFA")


def generate_config_path() -> str:
    """
    Generate the global configuration path for MFA

    Returns
    -------
    str
        Full path to configuration yaml
    """
    return os.path.join(TEMP_DIR, "global_config.yaml")


def generate_command_history_path() -> str:
    """
    Generate the path to the command history file

    Returns
    -------
    str
        Full path to history file
    """
    return os.path.join(TEMP_DIR, "command_history.yaml")


def load_command_history() -> List[str]:
    """
    Load command history for MFA

    Returns
    -------
    List
        List of commands previously run
    """
    path = generate_command_history_path()
    history = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf8") as f:
            history = yaml.safe_load(f)
    return history


def update_command_history(command_data: dict) -> None:
    """
    Update command history with most recent command

    Parameters
    ----------
    command_data: dict
        Current command metadata
    """
    try:
        if command_data["command"].split(" ")[1] == "history":
            return
    except Exception:
        return
    history = load_command_history()
    path = generate_command_history_path()
    history.append(command_data)
    history = history[-50:]
    with open(path, "w", encoding="utf8") as f:
        yaml.safe_dump(history, f)


def update_global_config(args: Namespace) -> None:
    """
    Update global configuration of MFA

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Arguments to set
    """
    global_configuration_file = generate_config_path()
    default_config = {
        "clean": False,
        "verbose": False,
        "debug": False,
        "overwrite": False,
        "terminal_colors": True,
        "terminal_width": 120,
        "cleanup_textgrids": True,
        "num_jobs": 3,
        "blas_num_threads": 1,
        "use_mp": True,
        "temp_directory": TEMP_DIR,
    }
    if os.path.exists(global_configuration_file):
        with open(global_configuration_file, "r", encoding="utf8") as f:
            data = yaml.safe_load(f)
            default_config.update(data)
    if args.always_clean:
        default_config["clean"] = True
    if args.never_clean:
        default_config["clean"] = False
    if args.always_verbose:
        default_config["verbose"] = True
    if args.never_verbose:
        default_config["verbose"] = False
    if args.always_debug:
        default_config["debug"] = True
    if args.never_debug:
        default_config["debug"] = False
    if args.always_overwrite:
        default_config["overwrite"] = True
    if args.never_overwrite:
        default_config["overwrite"] = False
    if args.disable_mp:
        default_config["use_mp"] = False
    if args.enable_mp:
        default_config["use_mp"] = True
    if args.disable_textgrid_cleanup:
        default_config["cleanup_textgrids"] = False
    if args.enable_textgrid_cleanup:
        default_config["cleanup_textgrids"] = True
    if args.disable_terminal_colors:
        default_config["terminal_colors"] = False
    if args.enable_terminal_colors:
        default_config["terminal_colors"] = True
    if args.num_jobs and args.num_jobs > 0:
        default_config["num_jobs"] = args.num_jobs
    if args.terminal_width and args.terminal_width > 0:
        default_config["terminal_width"] = args.terminal_width
    if args.blas_num_threads and args.blas_num_threads > 0:
        default_config["blas_num_threads"] = args.blas_num_threads
    if args.temp_directory:
        default_config["temp_directory"] = args.temp_directory
    with open(global_configuration_file, "w", encoding="utf8") as f:
        yaml.dump(default_config, f)


def load_global_config() -> Dict[str, Any]:
    """
    Load the global MFA configuration

    Returns
    -------
    Dict
        Global configuration
    """
    global_configuration_file = generate_config_path()
    default_config = {
        "clean": False,
        "verbose": False,
        "debug": False,
        "overwrite": False,
        "terminal_colors": True,
        "terminal_width": 120,
        "cleanup_textgrids": True,
        "num_jobs": 3,
        "blas_num_threads": 1,
        "use_mp": True,
        "temp_directory": TEMP_DIR,
    }
    if os.path.exists(global_configuration_file):
        with open(global_configuration_file, "r", encoding="utf8") as f:
            data = yaml.safe_load(f)
            default_config.update(data)
    return default_config


USE_COLORS = load_global_config().get("terminal_colors", True)
BLAS_THREADS = load_global_config().get("blas_num_threads", 1)
