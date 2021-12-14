"""
MFA configuration
=================

"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List

from montreal_forced_aligner.exceptions import RootDirectoryError

if TYPE_CHECKING:
    from argparse import Namespace

import os

import yaml

__all__ = [
    "generate_config_path",
    "generate_command_history_path",
    "load_command_history",
    "update_command_history",
    "update_global_config",
    "load_global_config",
    "USE_COLORS",
    "BLAS_THREADS",
]

MFA_ROOT_ENVIRONMENT_VARIABLE = "MFA_ROOT_DIR"


def get_temporary_directory():
    """
    Get the root temporary directory for MFA

    Returns
    -------
    str
        Root temporary directory

    Raises
    ------
    :class:`~montreal_forced_aligner.exceptions.RootDirectoryError`
    """
    TEMP_DIR = os.environ.get(MFA_ROOT_ENVIRONMENT_VARIABLE, os.path.expanduser("~/Documents/MFA"))
    try:
        os.makedirs(TEMP_DIR, exist_ok=True)
    except OSError:
        raise RootDirectoryError(TEMP_DIR, MFA_ROOT_ENVIRONMENT_VARIABLE)
    return TEMP_DIR


def generate_config_path() -> str:
    """
    Generate the global configuration path for MFA

    Returns
    -------
    str
        Full path to configuration yaml
    """
    return os.path.join(get_temporary_directory(), "global_config.yaml")


def generate_command_history_path() -> str:
    """
    Generate the path to the command history file

    Returns
    -------
    str
        Full path to history file
    """
    return os.path.join(get_temporary_directory(), "command_history.yaml")


def load_command_history() -> List[Dict[str, Any]]:
    """
    Load command history for MFA

    Returns
    -------
    list[dict[str, Any]]
        List of commands previously run
    """
    path = generate_command_history_path()
    history = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf8") as f:
            history = yaml.safe_load(f)
    for h in history:
        h["command"] = re.sub(r"^\S+.py ", "mfa ", h["command"])
    return history


def update_command_history(command_data: Dict[str, Any]) -> None:
    """
    Update command history with most recent command

    Parameters
    ----------
    command_data: dict[str, Any]
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
        "detect_phone_set": False,
        "num_jobs": 3,
        "blas_num_threads": 1,
        "use_mp": True,
        "temporary_directory": get_temporary_directory(),
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
    if args.disable_detect_phone_set:
        default_config["detect_phone_set"] = False
    if args.enable_detect_phone_set:
        default_config["detect_phone_set"] = True
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
    if args.temporary_directory:
        default_config["temporary_directory"] = args.temporary_directory
    with open(global_configuration_file, "w", encoding="utf8") as f:
        yaml.dump(default_config, f)


def load_global_config() -> Dict[str, Any]:
    """
    Load the global MFA configuration

    Returns
    -------
    dict[str, Any]
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
        "detect_phone_set": False,
        "num_jobs": 3,
        "blas_num_threads": 1,
        "use_mp": True,
        "temporary_directory": get_temporary_directory(),
    }
    if os.path.exists(global_configuration_file):
        with open(global_configuration_file, "r", encoding="utf8") as f:
            data = yaml.safe_load(f)
            default_config.update(data)
    if "temp_directory" in default_config:
        default_config["temporary_directory"] = default_config["temp_directory"]
    return default_config


USE_COLORS = load_global_config().get("terminal_colors", True)
BLAS_THREADS = load_global_config().get("blas_num_threads", 1)
