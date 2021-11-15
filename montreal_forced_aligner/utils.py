"""
Utility functions
=================

"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import textwrap
from typing import TYPE_CHECKING, Any, Dict, List, Union

import yaml
from colorama import Fore, Style

from .exceptions import KaldiProcessingError, ThirdpartyError
from .models import MODEL_TYPES

if TYPE_CHECKING:
    from .config.base_config import BaseConfig

__all__ = [
    "thirdparty_binary",
    "get_available_dictionaries",
    "log_config",
    "log_kaldi_errors",
    "get_available_models",
    "get_available_language_models",
    "get_available_acoustic_models",
    "get_available_g2p_models",
    "get_pretrained_language_model_path",
    "get_pretrained_g2p_path",
    "get_pretrained_ivector_path",
    "get_pretrained_path",
    "get_pretrained_acoustic_path",
    "get_dictionary_path",
    "get_available_ivector_extractors",
    "guess_model_type",
    "parse_logs",
    "setup_logger",
    "CustomFormatter",
]


def get_mfa_version():
    try:
        from .version import version as __version__  # noqa
    except ImportError:
        __version__ = "2.0.0"
    return __version__


def thirdparty_binary(binary_name: str) -> str:
    """
    Generate full path to a given binary name

    Notes
    -----
    With the move to conda, this function is deprecated as conda will manage the path much better

    Parameters
    ----------
    binary_name: str
        Executable to run

    Returns
    -------
    str
        Full path to the executable
    """
    bin_path = shutil.which(binary_name)
    if bin_path is None:
        if binary_name in ["fstcompile", "fstarcsort", "fstconvert"] and sys.platform != "win32":
            raise ThirdpartyError(binary_name, open_fst=True)
        else:
            raise ThirdpartyError(binary_name)
    return bin_path


def parse_logs(log_directory: str) -> None:
    """
    Parse the output of a Kaldi run for any errors and raise relevant MFA exceptions

    Parameters
    ----------
    log_directory: str
        Log directory to parse

    Raises
    ------
    KaldiProcessingError
        If any log files contained error lines

    """
    error_logs = []
    for name in os.listdir(log_directory):
        log_path = os.path.join(log_directory, name)
        with open(log_path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if "error while loading shared libraries: libopenblas.so.0" in line:
                    raise ThirdpartyError("libopenblas.so.0", open_blas=True)
                for libc_version in ["GLIBC_2.27", "GLIBCXX_3.4.20"]:
                    if libc_version in line:
                        raise ThirdpartyError(libc_version, libc=True)
                if "sox FAIL formats" in line:
                    f = line.split(" ")[-1]
                    raise ThirdpartyError(f, sox=True)
                if line.startswith("ERROR") or line.startswith("ASSERTION_FAILED"):
                    error_logs.append(log_path)
                    break
    if error_logs:
        raise KaldiProcessingError(error_logs)


def log_kaldi_errors(error_logs: List[str], logger: logging.Logger) -> None:
    """
    Save details of Kaldi processing errors to a logger

    Parameters
    ----------
    error_logs: List[str]
        Kaldi log files with errors
    logger: :class:`~logging.Logger`
        Logger to output to
    """
    logger.debug(f"There were {len(error_logs)} kaldi processing files that had errors:")
    for path in error_logs:
        logger.debug("")
        logger.debug(path)
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                logger.debug("\t" + line.strip())


def get_available_models(model_type: str) -> List[str]:
    """
    Get a list of available models for a given model type

    Parameters
    ----------
    model_type: str
        Model type to search

    Returns
    -------
    List[str]
        List of model names
    """
    from .config import TEMP_DIR

    pretrained_dir = os.path.join(TEMP_DIR, "pretrained_models", model_type)
    os.makedirs(pretrained_dir, exist_ok=True)
    available = []
    model_class = MODEL_TYPES[model_type]
    for f in os.listdir(pretrained_dir):
        if model_class is None or model_class.valid_extension(f):
            available.append(os.path.splitext(f)[0])
    return available


def guess_model_type(path: str) -> List[str]:
    """
    Guess a model type given a path

    Parameters
    ----------
    path: str
        Model archive to guess

    Returns
    -------
    List[str]
        Possible model types that use that extension
    """
    ext = os.path.splitext(path)[1]
    if not ext:
        return []
    possible = []
    for m, mc in MODEL_TYPES.items():
        if ext in mc.extensions:
            possible.append(m)
    return possible


def get_available_acoustic_models() -> List[str]:
    """
    Return a list of all available acoustic models

    Returns
    -------
    List[str]
        Pretrained acoustic models
    """
    return get_available_models("acoustic")


def get_available_g2p_models() -> List[str]:
    """
    Return a list of all available G2P models

    Returns
    -------
    List[str]
        Pretrained G2P models
    """
    return get_available_models("g2p")


def get_available_ivector_extractors() -> List[str]:
    """
    Return a list of all available ivector extractors

    Returns
    -------
    List[str]
        Pretrained ivector extractors
    """
    return get_available_models("ivector")


def get_available_language_models() -> List[str]:
    """
    Return a list of all available language models

    Returns
    -------
    List[str]
        Pretrained language models
    """
    return get_available_models("language_model")


def get_available_dictionaries() -> List[str]:
    """
    Return a list of all available dictionaries

    Returns
    -------
    List[str]
        Saved dictionaries
    """
    return get_available_models("dictionary")


def get_pretrained_path(model_type: str, name: str, enforce_existence: bool = True) -> str:
    """
    Generate a path to a pretrained model based on its name and model type

    Parameters
    ----------
    model_type: str
        Type of model
    name: str
        Name of model
    enforce_existence: bool
        Flag to return None if the path doesn't exist, defaults to True

    Returns
    -------
    str
        Path to model
    """
    from .config import TEMP_DIR

    pretrained_dir = os.path.join(TEMP_DIR, "pretrained_models", model_type)
    model_class = MODEL_TYPES[model_type]
    return model_class.generate_path(pretrained_dir, name, enforce_existence)


def get_pretrained_acoustic_path(name: str) -> str:
    """
    Generate a path to a given pretrained acoustic model

    Parameters
    ----------
    name: str
        Name of model

    Returns
    -------
    str
        Full path to model
    """
    return get_pretrained_path("acoustic", name)


def get_pretrained_ivector_path(name: str) -> str:
    """
    Generate a path to a given pretrained ivector extractor

    Parameters
    ----------
    name: str
        Name of model

    Returns
    -------
    str
        Full path to model
    """
    return get_pretrained_path("ivector", name)


def get_pretrained_language_model_path(name: str) -> str:
    """
    Generate a path to a given pretrained language model

    Parameters
    ----------
    name: str
        Name of model

    Returns
    -------
    str
        Full path to model
    """
    return get_pretrained_path("language_model", name)


def get_pretrained_g2p_path(name: str) -> str:
    """
    Generate a path to a given pretrained G2P model

    Parameters
    ----------
    name: str
        Name of model

    Returns
    -------
    str
        Full path to model
    """
    return get_pretrained_path("g2p", name)


def get_dictionary_path(name: str) -> str:
    """
    Generate a path to a given saved dictionary

    Parameters
    ----------
    name: str
        Name of dictionary

    Returns
    -------
    str
        Full path to dictionary
    """
    return get_pretrained_path("dictionary", name)


class CustomFormatter(logging.Formatter):
    """
    Custom log formatter class for MFA to highlight messages and incorporate terminal options from
    the global configuration
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .config import load_global_config

        config = load_global_config()
        self.width = config["terminal_width"]
        use_colors = config.get("terminal_colors", True)
        red = ""
        green = ""
        yellow = ""
        blue = ""
        reset = ""
        if use_colors:
            red = Fore.RED
            green = Fore.GREEN
            yellow = Fore.YELLOW
            blue = Fore.CYAN
            reset = Style.RESET_ALL

        self.FORMATS = {
            logging.DEBUG: (f"{blue}DEBUG{reset} - ", "%(message)s"),
            logging.INFO: (f"{green}INFO{reset} - ", "%(message)s"),
            logging.WARNING: (f"{yellow}WARNING{reset} - ", "%(message)s"),
            logging.ERROR: (f"{red}ERROR{reset} - ", "%(message)s"),
            logging.CRITICAL: (f"{red}CRITICAL{reset} - ", "%(message)s"),
        }

    def format(self, record: logging.LogRecord):
        """
        Format a given log message

        Parameters
        ----------
        record: logging.LogRecord
            Log record to format

        Returns
        -------
        str
            Formatted log message
        """
        log_fmt = self.FORMATS.get(record.levelno)
        return textwrap.fill(
            record.getMessage(),
            initial_indent=log_fmt[0],
            subsequent_indent=" " * len(log_fmt[0]),
            width=self.width,
        )


def setup_logger(
    identifier: str, output_directory: str, console_level: str = "info"
) -> logging.Logger:
    """
    Construct a logger for a command line run

    Parameters
    ----------
    identifier: str
        Name of the MFA utility
    output_directory: str
        Top level logging directory
    console_level: str, optional
        Level to output to the console, defaults to "info"

    Returns
    -------
    :class:`~logging.Logger`
        Logger to use
    """
    os.makedirs(output_directory, exist_ok=True)
    log_path = os.path.join(output_directory, f"{identifier}.log")
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = logging.getLogger(identifier)
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(log_path, encoding="utf8")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, console_level.upper()))
    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)
    logger.debug(f"Set up logger for MFA version: {get_mfa_version()}")
    return logger


def log_config(logger: logging.Logger, config: Union[Dict[str, Any], BaseConfig]) -> None:
    """
    Output a configuration to a Logger

    Parameters
    ----------
    logger: :class:`~logging.Logger`
        Logger to save to
    config: Dict[str, Any] or :class:`~montreal_forced_aligner.config.BaseConfig`
        Configuration to dump
    """
    stream = yaml.dump(config)
    logger.debug(stream)
