"""
Abstract Base Classes
=====================
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from abc import ABC, ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Type, Union, get_type_hints

import yaml

if TYPE_CHECKING:
    from argparse import Namespace

__all__ = [
    "MfaModel",
    "MfaWorker",
    "TopLevelMfaWorker",
    "MetaDict",
    "MappingType",
    "CtmErrorDict",
    "FileExporterMixin",
    "ModelExporterMixin",
    "TemporaryDirectoryMixin",
    "AdapterMixin",
    "TrainerMixin",
    "DictionaryEntryType",
    "ReversedMappingType",
    "Labels",
    "WordsType",
    "OneToOneMappingType",
    "OneToManyMappingType",
    "CorpusMappingType",
    "ScpType",
]

# Configuration types
MetaDict = dict[str, Any]
Labels: list[Any]
CtmErrorDict: dict[tuple[str, int], str]

# Dictionary types
DictionaryEntryType: list[dict[str, Union[tuple[str], float, None, int]]]
ReversedMappingType: dict[int, str]
WordsType: dict[str, DictionaryEntryType]
MappingType: dict[str, int]

# Corpus types
OneToOneMappingType: dict[str, str]
OneToManyMappingType: dict[str, list[str]]

CorpusMappingType: Union[OneToOneMappingType, OneToManyMappingType]
ScpType: Union[list[tuple[str, str]], list[tuple[str, list[Any]]]]


class TemporaryDirectoryMixin(metaclass=ABCMeta):
    """
    Abstract mixin class for MFA temporary directories

    Parameters
    ----------
    temporary_directory: str, optional
        Path to store temporary files
    """

    def __init__(
        self,
        temporary_directory: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not temporary_directory:
            from .config import get_temporary_directory

            temporary_directory = get_temporary_directory()
        self.temporary_directory = temporary_directory

    @property
    @abstractmethod
    def identifier(self) -> str:
        """Identifier to use in creating the temporary directory"""
        ...

    @property
    @abstractmethod
    def data_source_identifier(self) -> str:
        """Identifier for the data source (generally the corpus being used)"""
        ...

    @property
    @abstractmethod
    def output_directory(self) -> str:
        """Root temporary directory"""
        ...

    @property
    def corpus_output_directory(self) -> str:
        """Temporary directory containing all corpus information"""
        return os.path.join(self.output_directory, f"{self.data_source_identifier}")

    @property
    def dictionary_output_directory(self) -> str:
        """Temporary directory containing all dictionary information"""
        return os.path.join(self.output_directory, "dictionary")


class MfaWorker(metaclass=ABCMeta):
    """
    Abstract class for MFA workers

    Parameters
    ----------
    use_mp: bool
        Flag to run in multiprocessing mode, defaults to True
    debug: bool
        Flag to run in debug mode, defaults to False
    verbose: bool
        Flag to run in verbose mode, defaults to False

    Attributes
    ----------
    dirty: bool
        Flag for whether an error was encountered in processing
    """

    def __init__(
        self,
        use_mp: bool = True,
        debug: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.debug = debug
        self.verbose = verbose
        self.use_mp = use_mp
        self.dirty = False

    @abstractmethod
    def log_debug(self, message: str) -> None:
        """Abstract method for logging debug messages"""
        ...

    @abstractmethod
    def log_info(self, message: str) -> None:
        """Abstract method for logging info messages"""
        ...

    @abstractmethod
    def log_warning(self, message: str) -> None:
        """Abstract method for logging warning messages"""
        ...

    @abstractmethod
    def log_error(self, message: str) -> None:
        """Abstract method for logging error messages"""
        ...

    @classmethod
    def extract_relevant_parameters(cls, config: MetaDict) -> MetaDict:
        """
        Filter a configuration dictionary to just the relevant parameters for the current worker

        Parameters
        ----------
        config: dict[str, Any]
            Configuration dictionary

        Returns
        -------
        dict[str, Any]
            Filtered configuration dictionary
        """
        return {k: v for k, v in config.items() if k in cls.get_configuration_parameters()}

    @classmethod
    def get_configuration_parameters(cls) -> dict[str, Type]:
        """
        Get the types of parameters available to be configured

        Returns
        -------
        dict[str, Type]
            Dictionary of parameter names and their types
        """
        configuration_params = {}
        for t, ty in get_type_hints(cls.__init__).items():
            configuration_params[t] = ty
            try:
                if ty.__origin__ == Union:
                    configuration_params[t] = ty.__args__[0]
            except AttributeError:
                pass

        for c in cls.mro():
            try:
                for t, ty in get_type_hints(c.__init__).items():
                    configuration_params[t] = ty
                    try:
                        if ty.__origin__ == Union:
                            configuration_params[t] = ty.__args__[0]
                    except AttributeError:
                        pass
            except AttributeError:
                pass
        return configuration_params

    @property
    def configuration(self) -> MetaDict:
        """Configuration parameters"""
        return {
            "debug": self.debug,
            "verbose": self.verbose,
            "use_mp": self.use_mp,
            "dirty": self.dirty,
        }

    @property
    @abstractmethod
    def working_directory(self) -> str:
        """Current working directory"""
        ...

    @property
    def working_log_directory(self) -> str:
        """Current working log directory"""
        return os.path.join(self.working_directory, "log")

    @property
    @abstractmethod
    def data_directory(self) -> str:
        """Data directory"""
        ...


class TopLevelMfaWorker(MfaWorker, TemporaryDirectoryMixin, metaclass=ABCMeta):
    """
    Abstract mixin for top-level workers in MFA.  This class holds properties about the larger workflow run.

    Parameters
    ----------
    num_jobs: int
        Number of jobs and processes to uses
    clean: bool
        Flag for whether to remove any old files in the work directory
    """

    def __init__(
        self,
        num_jobs: int = 3,
        clean: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_jobs = num_jobs
        self.clean = clean
        self.initialized = False
        self.start_time = time.time()
        self.setup_logger()

    def __del__(self):
        """Ensure that loggers are cleaned up on delete"""
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    @abstractmethod
    def setup(self) -> None:
        """Abstract method for setting up a top-level worker"""
        ...

    @property
    def working_directory(self) -> str:
        """Alias for a folder that contains worker information, separate from the data directory"""
        return self.workflow_directory

    @classmethod
    def parse_args(cls, args: Optional[Namespace], unknown_args: Optional[list[str]]) -> MetaDict:
        """
        Class method for parsing configuration parameters from command line arguments

        Parameters
        ----------
        args: :class:`~argparse.Namespace`
            Arguments parsed by argparse
        unknown_args: list[str]
            Optional list of arguments that were not parsed by argparse

        Returns
        -------
        dict[str, Any]
            Dictionary of specified configuration parameters
        """
        param_types = cls.get_configuration_parameters()
        params = {}
        unknown_dict = {}
        if unknown_args:
            for i, a in enumerate(unknown_args):
                if not a.startswith("--"):
                    continue
                name = a.replace("--", "")
                if name not in param_types:
                    continue
                if i == len(unknown_args) - 1 or unknown_args[i + 1].startswith("--"):
                    val = True
                else:
                    val = unknown_args[i + 1]
                unknown_dict[name] = val
        for name, param_type in param_types.items():
            if name.endswith("_directory") or name.endswith("_path"):
                continue
            if args is not None and hasattr(args, name):
                params[name] = param_type(getattr(args, name))
            elif name in unknown_dict:
                params[name] = param_type(unknown_dict[name])
                if param_type == bool:
                    if unknown_dict[name].lower() == "false":
                        params[name] = False
        return params

    @classmethod
    def parse_parameters(
        cls,
        config_path: Optional[str] = None,
        args: Optional[Namespace] = None,
        unknown_args: Optional[list[str]] = None,
    ) -> MetaDict:
        """
        Parse configuration parameters from a config file and command line arguments

        Parameters
        ----------
        config_path: str, optional
            Path to yaml configuration file
        args: :class:`~argparse.Namespace`, optional
            Arguments parsed by argparse
        unknown_args: list[str], optional
            List of unknown arguments from argparse

        Returns
        -------
        dict[str, Any]
            Dictionary of specified configuration parameters
        """
        global_params = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf8") as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)
                for k, v in data.items():
                    global_params[k] = v
        global_params.update(cls.parse_args(args, unknown_args))
        return global_params

    @property
    @abstractmethod
    def workflow_identifier(self) -> str:
        """Identifier of the worker's workflow"""
        ...

    @property
    def worker_config_path(self):
        """Path to worker's configuration in the working directory"""
        return os.path.join(self.output_directory, f"{self.workflow_identifier}.yaml")

    def cleanup(self) -> None:
        """
        Clean up loggers and output final message for top-level workers
        """
        try:
            if self.dirty:
                self.logger.error("There was an error in the run, please see the log.")
            else:
                self.logger.info(f"Done! Everything took {time.time() - self.start_time} seconds")
            handlers = self.logger.handlers[:]
            for handler in handlers:
                handler.close()
                self.logger.removeHandler(handler)
            self.save_worker_config()
        except (NameError, ValueError):  # already cleaned up
            pass

    def save_worker_config(self):
        """Export worker configuration to its working directory"""
        with open(self.worker_config_path, "w") as f:
            yaml.dump(self.configuration, f)

    def _validate_previous_configuration(self, conf: MetaDict) -> bool:
        """
        Validate the current configuration against a previous configuration

        Parameters
        ----------
        conf: dict[str, Any]
            Previous run's configuration

        Returns
        -------
        bool
            Flag for whether the current run is compatible with the previous one
        """
        from montreal_forced_aligner.utils import get_mfa_version

        clean = True
        current_version = get_mfa_version()
        if conf["dirty"]:
            self.logger.debug("Previous run ended in an error (maybe ctrl-c?)")
            clean = False
        if "type" in conf:
            command = conf["type"]
        elif "command" in conf:
            command = conf["command"]
        else:
            command = self.workflow_identifier
        if command != self.workflow_identifier:
            self.logger.debug(
                f"Previous run was a different subcommand than {self.workflow_identifier} (was {command})"
            )
            clean = False
        if conf.get("version", current_version) != current_version:
            self.logger.debug(
                f"Previous run was on {conf['version']} version (new run: {current_version})"
            )
            clean = False
        for key in [
            "corpus_directory",
            "dictionary_path",
            "acoustic_model_path",
            "g2p_model_path",
            "language_model_path",
        ]:
            if conf.get(key, None) != getattr(self, key, None):
                self.logger.debug(
                    f"Previous run used a different {key.replace('_', ' ')} than {getattr(self, key, None)} (was {conf.get(key, None)})"
                )
                clean = False
        return clean

    def check_previous_run(self) -> bool:
        """
        Check whether a previous run has any conflicting settings with the current run.

        Returns
        -------
        bool
            Flag for whether the current run is compatible with the previous one
        """
        if not os.path.exists(self.worker_config_path):
            return True
        with open(self.worker_config_path, "r") as f:
            conf = yaml.load(f, Loader=yaml.SafeLoader)
        clean = self._validate_previous_configuration(conf)
        if not clean:
            self.logger.warning(
                "The previous run had a different configuration than the current, which may cause issues."
                " Please see the log for details or use --clean flag if issues are encountered."
            )
        return clean

    @property
    def identifier(self) -> str:
        """Combined identifier of the data source and workflow"""
        return f"{self.data_source_identifier}_{self.workflow_identifier}"

    @property
    def output_directory(self) -> str:
        """Root temporary directory to store all of this worker's files"""
        return os.path.join(self.temporary_directory, self.identifier)

    @property
    def workflow_directory(self) -> str:
        """Temporary directory to save work specific to the worker (i.e., not data)"""
        return os.path.join(self.output_directory, self.workflow_identifier)

    @property
    def log_file(self):
        """Path to the worker's log file"""
        return os.path.join(self.output_directory, f"{self.workflow_identifier}.log")

    def setup_logger(self):
        """
        Construct a logger for a command line run
        """
        from .utils import CustomFormatter, get_mfa_version

        if self.clean:
            shutil.rmtree(self.output_directory, ignore_errors=True)
        os.makedirs(self.workflow_directory, exist_ok=True)
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        self.logger = logging.getLogger(self.workflow_identifier)
        self.logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(self.log_file, encoding="utf8")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        handler = logging.StreamHandler(sys.stdout)
        if self.verbose:
            handler.setLevel(logging.DEBUG)
        else:
            handler.setLevel(logging.INFO)
        handler.setFormatter(CustomFormatter())
        self.logger.addHandler(handler)
        self.logger.debug(f"Set up logger for MFA version: {get_mfa_version()}")
        if self.clean:
            self.logger.debug("Cleaned previous run")

    def log_debug(self, message: str) -> None:
        """
        Log a debug message. This function is a wrapper around the :meth:`logging.Logger.debug`

        Parameters
        ----------
        message: str
            Debug message to log
        """
        self.logger.debug(message)

    def log_info(self, message: str) -> None:
        """
        Log an info message. This function is a wrapper around the :meth:`logging.Logger.info`

        Parameters
        ----------
        message: str
            Info message to log
        """
        self.logger.info(message)

    def log_warning(self, message: str) -> None:
        """
        Log a warning message. This function is a wrapper around the :meth:`logging.Logger.warning`

        Parameters
        ----------
        message: str
            Warning message to log
        """
        self.logger.warning(message)

    def log_error(self, message: str) -> None:
        """
        Log an error message. This function is a wrapper around the :meth:`logging.Logger.error`

        Parameters
        ----------
        message: str
            Error message to log
        """
        self.logger.error(message)


class ModelExporterMixin(metaclass=ABCMeta):
    """
    Abstract mixin class for exporting MFA models

    Parameters
    ----------
    overwrite: bool
        Flag for whether to overwrite the specified path if a file exists
    """

    def __init__(self, overwrite: bool = False, **kwargs):
        self.overwrite = overwrite
        super().__init__(**kwargs)

    @property
    @abstractmethod
    def meta(self) -> MetaDict:
        """Training configuration parameters"""
        ...

    @abstractmethod
    def export_model(self, output_model_path: str) -> None:
        """
        Abstract method to export an MFA model

        Parameters
        ----------
        output_model_path: str
            Path to export model
        """
        ...


class FileExporterMixin(metaclass=ABCMeta):
    """
    Abstract mixin class for exporting TextGrid and text files

    Parameters
    ----------
    overwrite: bool
        Flag for whether to overwrite files if they already exist

    """

    def __init__(self, overwrite: bool = False, cleanup_textgrids: bool = True, **kwargs):
        self.overwrite = overwrite
        self.cleanup_textgrids = cleanup_textgrids
        super().__init__(**kwargs)

    @property
    def backup_output_directory(self) -> Optional[str]:
        """Path to store files if overwriting is not allowed"""
        if self.overwrite:
            return None
        return os.path.join(self.working_directory, "backup")

    @abstractmethod
    def export_files(self, output_directory: str) -> None:
        """
        Export files to an output directory

        Parameters
        ----------
        output_directory: str
            Directory to export to
        """
        ...


class TrainerMixin(ModelExporterMixin):
    """
    Abstract mixin class for MFA trainers

    Parameters
    ----------
    num_iterations: int
        Number of training iterations

    Attributes
    ----------
    iteration: int
        Current iteration
    """

    def __init__(self, num_iterations: int = 40, **kwargs):
        super().__init__(**kwargs)
        self.iteration: int = 0
        self.num_iterations = num_iterations

    @abstractmethod
    def initialize_training(self) -> None:
        """Initialize training"""
        ...

    @abstractmethod
    def train(self) -> None:
        """Perform training"""
        ...

    @abstractmethod
    def train_iteration(self) -> None:
        """Run one training iteration"""
        ...

    @abstractmethod
    def finalize_training(self) -> None:
        """Finalize training"""
        ...


class AdapterMixin(ModelExporterMixin):
    """
    Abstract class for MFA model adaptation
    """

    @abstractmethod
    def adapt(self) -> None:
        """Perform adaptation"""
        ...


class MfaModel(ABC):
    """Abstract class for MFA models"""

    extensions: list[str]
    model_type = "base_model"

    @classmethod
    def pretrained_directory(cls) -> str:
        from .config import get_temporary_directory

        return os.path.join(get_temporary_directory(), "pretrained_models", cls.model_type)

    @classmethod
    def get_available_models(cls) -> list[str]:
        """
        Get a list of available models for a given model type

        Returns
        -------
        list[str]
            List of model names
        """
        if not os.path.exists(cls.pretrained_directory()):
            return []
        available = []
        for f in os.listdir(cls.pretrained_directory()):
            if cls.valid_extension(f):
                available.append(os.path.splitext(f)[0])
        return available

    @classmethod
    def get_pretrained_path(cls, name: str, enforce_existence: bool = True) -> str:
        """
        Generate a path to a pretrained model based on its name and model type

        Parameters
        ----------
        name: str
            Name of model
        enforce_existence: bool
            Flag to return None if the path doesn't exist, defaults to True

        Returns
        -------
        str
            Path to model
        """
        return cls.generate_path(cls.pretrained_directory(), name, enforce_existence)

    @classmethod
    @abstractmethod
    def valid_extension(cls, filename: str) -> bool:
        """Check whether a file has a valid extensions"""
        ...

    @classmethod
    @abstractmethod
    def generate_path(cls, root: str, name: str, enforce_existence: bool = True) -> Optional[str]:
        """Generate a path from a root directory"""
        ...

    @abstractmethod
    def pretty_print(self) -> None:
        """Print the model's meta data"""
        ...

    @property
    @abstractmethod
    def meta(self) -> MetaDict:
        """Meta data for the model"""
        ...

    @abstractmethod
    def add_meta_file(self, trainer: TrainerMixin) -> None:
        """Add meta data to the model"""
