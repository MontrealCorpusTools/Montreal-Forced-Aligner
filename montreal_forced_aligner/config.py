"""
MFA configuration
=================

"""
from __future__ import annotations

import os
import pathlib
import re
import typing
from typing import Any, Dict, List, Union

import dataclassy
import joblib
import rich_click as click
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
    "MFA_ROOT_ENVIRONMENT_VARIABLE",
    "MFA_PROFILE_VARIABLE",
    "IVECTOR_DIMENSION",
    "XVECTOR_DIMENSION",
    "PLDA_DIMENSION",
    "MEMORY",
]

MFA_ROOT_ENVIRONMENT_VARIABLE = "MFA_ROOT_DIR"
MFA_PROFILE_VARIABLE = "MFA_PROFILE"

IVECTOR_DIMENSION = 192
XVECTOR_DIMENSION = 192
PLDA_DIMENSION = 192


def get_temporary_directory() -> pathlib.Path:
    """
    Get the root temporary directory for MFA

    Returns
    -------
    Path
        Root temporary directory

    Raises
    ------
        :class:`~montreal_forced_aligner.exceptions.RootDirectoryError`
    """
    TEMP_DIR = pathlib.Path(
        os.environ.get(MFA_ROOT_ENVIRONMENT_VARIABLE, "~/Documents/MFA")
    ).expanduser()
    try:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise RootDirectoryError(TEMP_DIR, MFA_ROOT_ENVIRONMENT_VARIABLE)
    return TEMP_DIR


def generate_config_path() -> pathlib.Path:
    """
    Generate the global configuration path for MFA

    Returns
    -------
    Path
        Full path to configuration yaml
    """
    return get_temporary_directory().joinpath("global_config.yaml")


def generate_command_history_path() -> pathlib.Path:
    """
    Generate the path to the command history file

    Returns
    -------
    Path
        Full path to history file
    """
    return get_temporary_directory().joinpath("command_history.yaml")


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
    if path.exists():
        with mfa_open(path, "r") as f:
            history = yaml.load(f, Loader=yaml.Loader)
            if not history:
                history = []
    history = [h for h in history if h["command"]]
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
        yaml.dump(history, f, Dumper=yaml.Dumper, allow_unicode=True)


CLEAN = False
FINAL_CLEAN = False
VERBOSE = False
DEBUG = False
QUIET = False
OVERWRITE = False
CLEANUP_TEXTGRIDS = True
USE_POSTGRES = False
SEED = 1234
NUM_JOBS = 3
USE_MP = True
USE_THREADING = False
SINGLE_SPEAKER = False
DATABASE_LIMITED_MODE = False
AUTO_SERVER = True
TEMPORARY_DIRECTORY = get_temporary_directory()
GITHUB_TOKEN = None
HF_TOKEN = None
BLAS_NUM_THREADS = 1
BYTES_LIMIT = 100e6
CURRENT_PROFILE_NAME = os.getenv(MFA_PROFILE_VARIABLE, "global")


def database_socket() -> str:
    p = get_temporary_directory().joinpath(f"pg_mfa_{CURRENT_PROFILE_NAME}_socket")
    p.mkdir(parents=True, exist_ok=True)
    return p.as_posix()


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
    cleanup_textgrids: bool = True
    use_postgres: bool = False
    database_limited_mode: bool = False
    bytes_limit: int = 100e6
    seed: int = 0
    num_jobs: int = 3
    blas_num_threads: int = 1
    use_mp: bool = True
    use_threading: bool = True
    single_speaker: bool = False
    auto_server: bool = True
    temporary_directory: pathlib.Path = get_temporary_directory()
    github_token: typing.Optional[str] = None
    hf_token: typing.Optional[str] = None

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
            if k == "temporary_directory":
                v = pathlib.Path(v)
            if hasattr(self, k):
                setattr(self, k, v)


class MfaConfiguration:
    """
    Global MFA configuration class
    """

    def __init__(self):
        self.current_profile_name = CURRENT_PROFILE_NAME
        self.config_path = generate_config_path()
        self.global_profile = MfaProfile()
        self.profiles: Dict[str, MfaProfile] = {"global": self.global_profile}
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
    def database_socket(self) -> str:
        p = get_temporary_directory().joinpath(f"pg_mfa_{self.current_profile_name}_socket")
        p.mkdir(parents=True, exist_ok=True)
        return p.as_posix()

    @property
    def current_profile(self) -> MfaProfile:
        """Name of the current :class:`~montreal_forced_aligner.config.MfaProfile`"""
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
            data = yaml.load(f, Loader=yaml.Loader)
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


def load_configuration():
    config_path = generate_config_path()
    if not config_path.exists():
        return
    with mfa_open(config_path, "r") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    profiles = data.pop("profiles", {})
    if CURRENT_PROFILE_NAME == "global":
        update_configuration(data)
    else:
        for name, p in profiles.items():
            if name == CURRENT_PROFILE_NAME:
                update_configuration(p)


def update_configuration(data):
    for k, v in data.items():
        k = k.upper()
        if k in globals():
            if k == "TEMPORARY_DIRECTORY":
                v = pathlib.Path(v)
            if v is None:
                continue
            globals()[k] = v


GLOBAL_CONFIG = MfaConfiguration()
MEMORY = joblib.Memory(
    location=os.path.join(get_temporary_directory(), "joblib_cache"),
    verbose=4 if VERBOSE else 0,
    bytes_limit=BYTES_LIMIT,
)

os.environ["OMP_NUM_THREADS"] = f"{BLAS_NUM_THREADS}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{BLAS_NUM_THREADS}"
os.environ["MKL_NUM_THREADS"] = f"{BLAS_NUM_THREADS}"
