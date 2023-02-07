"""
MFA configuration
=================

"""
from __future__ import annotations

import os
import re
import typing
from typing import Any, Dict, List, Union

import click
import dataclassy
import joblib
import yaml
from dataclassy import dataclass

from montreal_forced_aligner.exceptions import RootDirectoryError
from montreal_forced_aligner.helper import mfa_open

__all__ = [
    "generate_config_path",
    "generate_command_history_path",
    "load_command_history",
    "get_temporary_directory",
    "update_command_history",
    "MfaConfiguration",
    "GLOBAL_CONFIG",
]

MFA_ROOT_ENVIRONMENT_VARIABLE = "MFA_ROOT_DIR"
MFA_PROFILE_VARIABLE = "MFA_PROFILE"

IVECTOR_DIMENSION = 192
XVECTOR_DIMENSION = 192
PLDA_DIMENSION = 192


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
        with mfa_open(path, "r") as f:
            history = yaml.safe_load(f)
            if not history:
                history = []
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
    with mfa_open(path, "w") as f:
        yaml.safe_dump(history, f, allow_unicode=True)


@dataclass(slots=True)
class MfaProfile:
    """
    Configuration class for a profile used from the command line
    """

    clean: bool = False
    verbose: bool = False
    debug: bool = False
    quiet: bool = False
    overwrite: bool = False
    terminal_colors: bool = True
    cleanup_textgrids: bool = True
    database_backend: str = "psycopg2"
    database_port: int = 5433
    bytes_limit: int = 100e6
    seed: int = 0
    num_jobs: int = 3
    blas_num_threads: int = 1
    use_mp: bool = True
    single_speaker: bool = False
    temporary_directory: str = get_temporary_directory()
    github_token: typing.Optional[str] = None

    def __getitem__(self, item):
        """Get key from profile"""
        return getattr(self, item)

    def update(self, data: Union[Dict[str, Any], click.Context]) -> None:
        """
        Update configuration from new data

        Parameters
        ----------
        data: typing.Union[dict[str, typing.Any], :class:`click.Context`]
            Parameters to update
        """
        for k, v in data.items():
            if k == "temp_directory":
                k = "temporary_directory"
            if v is None:
                continue
            if hasattr(self, k):
                setattr(self, k, v)


class MfaConfiguration:
    """
    Global MFA configuration class
    """

    def __init__(self):
        self.current_profile_name = os.getenv(MFA_PROFILE_VARIABLE, "global")
        self.config_path = generate_config_path()
        self.global_profile = MfaProfile()
        self.profiles: Dict[str, MfaProfile] = {}
        self.profiles["global"] = self.global_profile
        if not os.path.exists(self.config_path):
            self.save()
        else:
            self.load()

    def __getattr__(self, item):
        """Get key from current profile"""
        if hasattr(self.current_profile, item):
            return getattr(self.current_profile, item)

    def __getitem__(self, item):
        """Get key from current profile"""
        if hasattr(self.current_profile, item):
            return getattr(self.current_profile, item)

    @property
    def current_profile(self) -> MfaProfile:
        """Name of the current :class:`~montreal_forced_aligner.config.MfaProfile`"""
        self.current_profile_name = os.getenv(MFA_PROFILE_VARIABLE, "global")
        if self.current_profile_name not in self.profiles:
            self.profiles[self.current_profile_name] = MfaProfile()
            self.profiles[self.current_profile_name].update(dataclassy.asdict(self.global_profile))
        return self.profiles[self.current_profile_name]

    def save(self) -> None:
        """Save MFA configuration"""
        global_configuration_file = generate_config_path()
        data = dataclassy.asdict(self.global_profile)
        data["profiles"] = {
            k: dataclassy.asdict(v) for k, v in self.profiles.items() if k != "global"
        }
        with mfa_open(global_configuration_file, "w") as f:
            yaml.dump(data, f)

    def load(self) -> None:
        """Load MFA configuration"""
        with mfa_open(self.config_path, "r") as f:
            data = yaml.safe_load(f)
        for name, p in data.pop("profiles", {}).items():
            self.profiles[name] = MfaProfile()
            self.profiles[name].update(p)
        self.global_profile.update(data)
        if (
            self.current_profile_name not in self.profiles
            and self.current_profile_name != "global"
        ):
            self.profiles[self.current_profile_name] = MfaProfile()
            self.profiles[self.current_profile_name].update(data)


GLOBAL_CONFIG = MfaConfiguration()
MEMORY = joblib.Memory(
    location=os.path.join(get_temporary_directory(), "joblib_cache"),
    verbose=4 if GLOBAL_CONFIG.current_profile.verbose else 0,
    bytes_limit=GLOBAL_CONFIG.current_profile.bytes_limit,
)
