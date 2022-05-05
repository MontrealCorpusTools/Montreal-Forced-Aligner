"""
Abstract Base Classes
=====================
"""

from __future__ import annotations

import abc
import logging
import os
import shutil
import subprocess
import sys
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    get_type_hints,
)

import sqlalchemy
import yaml
from sqlalchemy.orm import Session

from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.helper import comma_join, load_configuration

if TYPE_CHECKING:
    from argparse import Namespace

    from montreal_forced_aligner.data import MfaArguments

__all__ = [
    "MfaModel",
    "MfaWorker",
    "TopLevelMfaWorker",
    "MetaDict",
    "FileExporterMixin",
    "ModelExporterMixin",
    "TemporaryDirectoryMixin",
    "AdapterMixin",
    "TrainerMixin",
    "KaldiFunction",
]

# Configuration types
MetaDict = Dict[str, Any]


class KaldiFunction(metaclass=abc.ABCMeta):
    """
    Abstract class for running Kaldi functions
    """

    def __init__(self, args: MfaArguments):
        self.args = args
        self.db_path = self.args.db_path
        self.job_name = self.args.job_name
        self.log_path = self.args.log_path

    @abc.abstractmethod
    def run(self):
        """Run the function"""
        ...

    def check_call(self, proc: subprocess.Popen):
        """
        Check whether a subprocess successfully completed

        Parameters
        ----------
        proc: subprocess.Popen
            Subprocess to check

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there was an error running the subprocess
        """
        if proc.returncode is None:
            proc.wait()
        if proc.returncode != 0:
            raise KaldiProcessingError([self.log_path])


class TemporaryDirectoryMixin(metaclass=abc.ABCMeta):
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
        self._corpus_output_directory = None
        self._dictionary_output_directory = None
        self._language_model_output_directory = None

    @property
    @abc.abstractmethod
    def identifier(self) -> str:
        """Identifier to use in creating the temporary directory"""
        ...

    @property
    @abc.abstractmethod
    def data_source_identifier(self) -> str:
        """Identifier for the data source (generally the corpus being used)"""
        ...

    @property
    @abc.abstractmethod
    def output_directory(self) -> str:
        """Root temporary directory"""
        ...

    @property
    def corpus_output_directory(self) -> str:
        """Temporary directory containing all corpus information"""
        if self._corpus_output_directory:
            return self._corpus_output_directory
        return os.path.join(self.output_directory, f"{self.data_source_identifier}")

    @corpus_output_directory.setter
    def corpus_output_directory(self, directory: str) -> None:
        self._corpus_output_directory = directory

    @property
    def dictionary_output_directory(self) -> str:
        """Temporary directory containing all dictionary information"""
        if self._dictionary_output_directory:
            return self._dictionary_output_directory
        return os.path.join(self.output_directory, "dictionary")

    @dictionary_output_directory.setter
    def dictionary_output_directory(self, directory: str) -> None:
        self._dictionary_output_directory = directory

    @property
    def language_model_output_directory(self) -> str:
        """Temporary directory containing all dictionary information"""
        if self._language_model_output_directory:
            return self._language_model_output_directory
        return os.path.join(self.output_directory, "language_model")

    @language_model_output_directory.setter
    def language_model_output_directory(self, directory: str) -> None:
        self._language_model_output_directory = directory


class DatabaseMixin(TemporaryDirectoryMixin, metaclass=abc.ABCMeta):
    """
    Abstract class for mixing in database functionality
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._db_engine = None

    def initialize_database(self) -> None:
        """
        Initialize the database with database schema
        """
        from montreal_forced_aligner.db import MfaSqlBase

        os.makedirs(self.output_directory, exist_ok=True)
        MfaSqlBase.metadata.create_all(self.db_engine)

    @property
    def db_engine(self) -> sqlalchemy.engine.Engine:
        """Database engine"""
        if self._db_engine is None:
            self._db_engine = self.construct_engine()
        return self._db_engine

    @property
    def db_path(self) -> str:
        """Path to SQLite database file"""
        return os.path.join(self.output_directory, f"{self.identifier}.db")

    def construct_engine(self, same_thread=True, read_only=False) -> sqlalchemy.engine.Engine:
        """
        Construct a database engine

        Parameters
        ----------
        same_thread: bool, optional
            Flag for whether to enforce checking access on different threads, defaults to True
        read_only: bool, optional
            Flag for whether the database engine should be created as read-only, defaults to False

        Returns
        -------
        :class:`~sqlalchemy.engine.Engine`
            SqlAlchemy engine
        """
        connect_args = {}
        if not same_thread:
            connect_args["check_same_thread"] = False
        string = f"sqlite:///{self.db_path}"
        if read_only:
            string = f"sqlite:///file:{self.db_path}?mode=ro&nolock=1&uri=true"
        return sqlalchemy.create_engine(string, connect_args=connect_args)

    def session(self, **kwargs) -> Session:
        """
        Construct database session

        Parameters
        ----------
        **kwargs
            Keyword arguments to pass to the Session

        Returns
        -------
        :class:`~sqlalchemy.orm.session.Session`
            SqlAlchemy session
        """
        autoflush = kwargs.pop("autoflush", False)
        return sqlalchemy.orm.Session(self.db_engine, autoflush=autoflush, **kwargs)


class MfaWorker(metaclass=abc.ABCMeta):
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
    quiet: bool
        Flag for whether to suppress printing to the terminal

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
        quiet: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.debug = debug
        self.verbose = verbose
        self.use_mp = use_mp
        self.dirty = False
        self.quiet = quiet

    def log_debug(self, message: str = "") -> None:
        """
        Print a debug message

        Parameters
        ----------
        message: str
            Debug message to log
        """
        if not self.quiet and self.verbose:
            print(message)

    def log_error(self, message: str = "") -> None:
        """
        Print an error message

        Parameters
        ----------
        message: str
            Error message to log
        """
        if not self.quiet:
            print(message)

    def log_info(self, message: str = "") -> None:
        """
        Print an info message

        Parameters
        ----------
        message: str
            Info message to log
        """
        if not self.quiet:
            print(message)

    def log_warning(self, message: str = "") -> None:
        """
        Print a warning message

        Parameters
        ----------
        message: str
            Warning message to log
        """
        if not self.quiet:
            print(message)

    @classmethod
    def extract_relevant_parameters(cls, config: MetaDict) -> Tuple[MetaDict, List[str]]:
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
        list[str]
            Skipped keys
        """
        skipped = []
        new_config = {}
        for k, v in config.items():
            if k in cls.get_configuration_parameters():
                new_config[k] = v
            else:
                skipped.append(k)
        return new_config, skipped

    @classmethod
    def get_configuration_parameters(cls) -> Dict[str, Type]:
        """
        Get the types of parameters available to be configured

        Returns
        -------
        dict[str, Type]
            Dictionary of parameter names and their types
        """
        mapping = {Dict: dict, Tuple: tuple, List: list, Set: set}
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
        for t, ty in configuration_params.items():
            for v in mapping.values():
                try:
                    if ty.__origin__ == v:
                        configuration_params[t] = v
                        break
                except AttributeError:
                    break
        return configuration_params

    @property
    def configuration(self) -> MetaDict:
        """Configuration parameters"""
        return {
            "debug": self.debug,
            "verbose": self.verbose,
            "quiet": self.quiet,
            "use_mp": self.use_mp,
            "dirty": self.dirty,
        }

    @property
    @abc.abstractmethod
    def working_directory(self) -> str:
        """Current working directory"""
        ...

    @property
    def working_log_directory(self) -> str:
        """Current working log directory"""
        return os.path.join(self.working_directory, "log")

    @property
    @abc.abstractmethod
    def data_directory(self) -> str:
        """Data directory"""
        ...


class TopLevelMfaWorker(MfaWorker, TemporaryDirectoryMixin, metaclass=abc.ABCMeta):
    """
    Abstract mixin for top-level workers in MFA.  This class holds properties about the larger workflow run.

    Parameters
    ----------
    num_jobs: int
        Number of jobs and processes to use
    clean: bool
        Flag for whether to remove any old files in the work directory
    """

    nullable_fields = [
        "punctuation",
        "compound_markers",
        "clitic_markers",
        "quote_markers",
        "word_break_markers",
    ]

    def __init__(
        self,
        num_jobs: int = 3,
        clean: bool = False,
        **kwargs,
    ):
        kwargs, skipped = type(self).extract_relevant_parameters(kwargs)
        super().__init__(**kwargs)
        self.num_jobs = num_jobs
        self.clean = clean
        self.initialized = False
        self.start_time = time.time()
        self.setup_logger()
        if skipped:
            self.log_warning(f"Skipped the following configuration keys: {comma_join(skipped)}")

    def __del__(self):
        """Ensure that loggers are cleaned up on delete"""
        logger = logging.getLogger(self.identifier)
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)

    @abc.abstractmethod
    def setup(self) -> None:
        """Abstract method for setting up a top-level worker"""
        ...

    @property
    def working_directory(self) -> str:
        """Alias for a folder that contains worker information, separate from the data directory"""
        return self.workflow_directory

    @classmethod
    def parse_args(cls, args: Optional[Namespace], unknown_args: Optional[List[str]]) -> MetaDict:
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
            if (name.endswith("_directory") and name != "audio_directory") or name.endswith(
                "_path"
            ):
                continue
            if args is not None and hasattr(args, name) and getattr(args, name) is not None:
                params[name] = param_type(getattr(args, name))
            elif name in unknown_dict:
                params[name] = param_type(unknown_dict[name])
                if param_type == bool:
                    if unknown_dict[name].lower() == "false":
                        params[name] = False
        if getattr(args, "disable_mp", False):
            params["use_mp"] = False
        elif getattr(args, "disable_textgrid_cleanup", False):
            params["cleanup_textgrids"] = False
        return params

    @classmethod
    def parse_parameters(
        cls,
        config_path: Optional[str] = None,
        args: Optional[Namespace] = None,
        unknown_args: Optional[List[str]] = None,
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
            data = load_configuration(config_path)
            for k, v in data.items():
                if v is None and k in cls.nullable_fields:
                    v = []
                global_params[k] = v
        global_params.update(cls.parse_args(args, unknown_args))
        return global_params

    @property
    @abc.abstractmethod
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
            if getattr(self, "_session", None) is not None:
                try:
                    self._session.commit()
                except Exception:
                    self._session.rollback()
                finally:
                    self._session.close()
                self._session = None
            if getattr(self, "_db_engine", None) is not None:
                self._db_engine.dispose()
                self._db_engine = None
            if self.dirty:
                self.log_error("There was an error in the run, please see the log.")
            else:
                self.log_info(f"Done! Everything took {time.time() - self.start_time} seconds")
            logger = logging.getLogger(self.identifier)
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)
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
            self.log_debug("Previous run ended in an error (maybe ctrl-c?)")
            clean = False
        if "type" in conf:
            command = conf["type"]
        elif "command" in conf:
            command = conf["command"]
        else:
            command = self.workflow_identifier
        if command != self.workflow_identifier:
            self.log_debug(
                f"Previous run was a different subcommand than {self.workflow_identifier} (was {command})"
            )
            clean = False
        if conf.get("version", current_version) != current_version:
            self.log_debug(
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
                self.log_debug(
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
        conf = load_configuration(self.worker_config_path)
        clean = self._validate_previous_configuration(conf)
        if not clean:
            self.log_warning(
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

        current_version = get_mfa_version()
        # Remove previous directory if versions are different
        if os.path.exists(self.worker_config_path):
            conf = load_configuration(self.worker_config_path)
            if conf.get("version", current_version) != current_version:
                self.clean = True
        if self.clean:
            shutil.rmtree(self.output_directory, ignore_errors=True)
        os.makedirs(self.workflow_directory, exist_ok=True)
        logger = logging.getLogger(self.identifier)
        logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(self.log_file, encoding="utf8")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        if not self.quiet:
            handler = logging.StreamHandler(sys.stdout)
            if self.verbose:
                handler.setLevel(logging.DEBUG)
            else:
                handler.setLevel(logging.INFO)
            handler.setFormatter(CustomFormatter())
            logger.addHandler(handler)
        logger.debug(
            f"Beginning run for {self.workflow_identifier} on {self.data_source_identifier}"
        )
        if self.use_mp:
            logger.debug(f"Using multiprocessing with {self.num_jobs}")
        else:
            logger.debug(f"NOT using multiprocessing with {self.num_jobs}")
        logger.debug(f"Set up logger for MFA version: {current_version}")
        if self.clean:
            logger.debug("Cleaned previous run")

    def log_debug(self, message: str = "") -> None:
        """
        Log a debug message. This function is a wrapper around the :meth:`logging.Logger.debug`

        Parameters
        ----------
        message: str
            Debug message to log
        """
        logger = logging.getLogger(self.identifier)
        logger.debug(message)

    def log_info(self, message: str = "") -> None:
        """
        Log an info message. This function is a wrapper around the :meth:`logging.Logger.info`

        Parameters
        ----------
        message: str
            Info message to log
        """
        logger = logging.getLogger(self.identifier)
        logger.info(message)

    def log_warning(self, message: str = "") -> None:
        """
        Log a warning message. This function is a wrapper around the :meth:`logging.Logger.warning`

        Parameters
        ----------
        message: str
            Warning message to log
        """
        logger = logging.getLogger(self.identifier)
        logger.warning(message)

    def log_error(self, message: str = "") -> None:
        """
        Log an error message. This function is a wrapper around the :meth:`logging.Logger.error`

        Parameters
        ----------
        message: str
            Error message to log
        """
        logger = logging.getLogger(self.identifier)
        logger.error(message)


class ExporterMixin(metaclass=abc.ABCMeta):
    """
    Abstract mixin class for exporting any kind of file

    Parameters
    ----------
    overwrite: bool
        Flag for whether to overwrite the specified path if a file exists
    """

    def __init__(self, overwrite: bool = False, **kwargs):
        self.overwrite = overwrite
        super().__init__(**kwargs)


class ModelExporterMixin(ExporterMixin, metaclass=abc.ABCMeta):
    """
    Abstract mixin class for exporting MFA models
    """

    @property
    @abc.abstractmethod
    def meta(self) -> MetaDict:
        """Training configuration parameters"""
        ...

    @abc.abstractmethod
    def export_model(self, output_model_path: str) -> None:
        """
        Abstract method to export an MFA model

        Parameters
        ----------
        output_model_path: str
            Path to export model
        """
        ...


class FileExporterMixin(ExporterMixin, metaclass=abc.ABCMeta):
    """
    Abstract mixin class for exporting TextGrid and text files

    Parameters
    ----------
    cleanup_textgrids: bool
        Flag for whether to clean up exported TextGrids
    """

    def __init__(self, cleanup_textgrids: bool = True, **kwargs):
        self.cleanup_textgrids = cleanup_textgrids
        super().__init__(**kwargs)

    @abc.abstractmethod
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

    @abc.abstractmethod
    def initialize_training(self) -> None:
        """Initialize training"""
        ...

    @abc.abstractmethod
    def train(self) -> None:
        """Perform training"""
        ...

    @abc.abstractmethod
    def train_iteration(self) -> None:
        """Run one training iteration"""
        ...

    @abc.abstractmethod
    def finalize_training(self) -> None:
        """Finalize training"""
        ...


class AdapterMixin(ModelExporterMixin):
    """
    Abstract class for MFA model adaptation
    """

    @abc.abstractmethod
    def adapt(self) -> None:
        """Perform adaptation"""
        ...


class MfaModel(abc.ABC):
    """Abstract class for MFA models"""

    extensions: List[str]
    model_type = "base_model"

    @classmethod
    def pretrained_directory(cls) -> str:
        """Directory that pretrained models are saved in"""
        from .config import get_temporary_directory

        path = os.path.join(get_temporary_directory(), "pretrained_models", cls.model_type)
        os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def get_available_models(cls) -> List[str]:
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
    @abc.abstractmethod
    def valid_extension(cls, filename: str) -> bool:
        """Check whether a file has a valid extensions"""
        ...

    @classmethod
    @abc.abstractmethod
    def generate_path(cls, root: str, name: str, enforce_existence: bool = True) -> Optional[str]:
        """Generate a path from a root directory"""
        ...

    @abc.abstractmethod
    def pretty_print(self) -> None:
        """Print the model's meta data"""
        ...

    @property
    @abc.abstractmethod
    def meta(self) -> MetaDict:
        """Metadata for the model"""
        ...

    @abc.abstractmethod
    def add_meta_file(self, trainer: TrainerMixin) -> None:
        """Add metadata to the model"""
