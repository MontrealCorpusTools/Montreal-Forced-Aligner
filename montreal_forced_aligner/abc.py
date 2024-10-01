"""
Abstract Base Classes
=====================
"""

from __future__ import annotations

import abc
import contextlib
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import typing
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

import requests
import sqlalchemy
import yaml
from sqlalchemy.orm import scoped_session, sessionmaker

from montreal_forced_aligner import config
from montreal_forced_aligner.db import CorpusWorkflow, MfaSqlBase
from montreal_forced_aligner.exceptions import (
    DatabaseError,
    KaldiProcessingError,
    MultiprocessingError,
)
from montreal_forced_aligner.helper import comma_join, load_configuration, mfa_open

if TYPE_CHECKING:
    from pathlib import Path

    from montreal_forced_aligner.data import MfaArguments, WorkflowType

__all__ = [
    "MfaModel",
    "MfaWorker",
    "TopLevelMfaWorker",
    "MetaDict",
    "DatabaseMixin",
    "FileExporterMixin",
    "ModelExporterMixin",
    "TemporaryDirectoryMixin",
    "AdapterMixin",
    "TrainerMixin",
    "KaldiFunction",
]

# Configuration types
MetaDict = Dict[str, Any]
logger = logging.getLogger("mfa")


class KaldiFunction(metaclass=abc.ABCMeta):
    """
    Abstract class for running Kaldi functions
    """

    def __init__(self, args: MfaArguments):
        self.args = args
        self.db_string = None
        self._session = None
        if isinstance(self.args.session, str):
            self.db_string = self.args.session
        else:
            self._session = self.args.session
        self.job_name = self.args.job_name
        self.log_path = self.args.log_path
        self.callback = None

    @contextlib.contextmanager
    def session(self):
        if self._session is not None:
            with self._session() as session:
                yield session
        else:
            db_engine = sqlalchemy.create_engine(self.db_string)
            with sqlalchemy.orm.Session(db_engine) as session:
                yield session

    def run(self):
        """Run the function, calls subclassed object's ``_run`` with error handling"""
        try:
            if self._session is not None:
                config.USE_THREADING = True
            else:
                config.USE_THREADING = False
            self._run()
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            error_text = "\n".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            raise MultiprocessingError(self.job_name, error_text)

    def _run(self) -> None:
        """Internal logic for running the worker"""
        pass

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
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._corpus_output_directory = None
        self._dictionary_output_directory = None
        self._language_model_output_directory = None
        self._acoustic_model_output_directory = None
        self._g2p_model_output_directory = None
        self._ivector_extractor_output_directory = None
        self._current_workflow = None

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
    def output_directory(self) -> Path:
        """Root temporary directory"""
        ...

    def clean_working_directory(self) -> None:
        """Clean up previous runs"""
        shutil.rmtree(self.output_directory, ignore_errors=True)

    @property
    def corpus_output_directory(self) -> Path:
        """Temporary directory containing all corpus information"""
        if self._corpus_output_directory:
            return self._corpus_output_directory
        return self.output_directory.joinpath(f"{self.data_source_identifier}")

    @corpus_output_directory.setter
    def corpus_output_directory(self, directory: Path) -> None:
        self._corpus_output_directory = directory

    @property
    def dictionary_output_directory(self) -> Path:
        """Temporary directory containing all dictionary information"""
        if self._dictionary_output_directory:
            return self._dictionary_output_directory
        return self.output_directory.joinpath("dictionary")

    @property
    def model_output_directory(self) -> Path:
        """Temporary directory containing all dictionary information"""
        return self.output_directory.joinpath("models")

    @dictionary_output_directory.setter
    def dictionary_output_directory(self, directory: Path) -> None:
        self._dictionary_output_directory = directory

    @property
    def language_model_output_directory(self) -> Path:
        """Temporary directory containing all dictionary information"""
        if self._language_model_output_directory:
            return self._language_model_output_directory
        return self.model_output_directory.joinpath("language_model")

    @language_model_output_directory.setter
    def language_model_output_directory(self, directory: Path) -> None:
        self._language_model_output_directory = directory

    @property
    def acoustic_model_output_directory(self) -> Path:
        """Temporary directory containing all dictionary information"""
        if self._acoustic_model_output_directory:
            return self._acoustic_model_output_directory
        return self.model_output_directory.joinpath("acoustic_model")

    @acoustic_model_output_directory.setter
    def acoustic_model_output_directory(self, directory: Path) -> None:
        self._acoustic_model_output_directory = directory


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
        self._db_path = None
        self._session = None
        self.database_initialized = False

    def cleanup_connections(self) -> None:
        if getattr(self, "_session", None) is not None:
            self._session.remove()
            del self._session
            self._session = None

        if getattr(self, "_db_engine", None) is not None:
            self._db_engine.dispose()
            del self._db_engine
            self._db_engine = None

    def delete_database(self) -> None:
        """
        Reset all schemas
        """

        if config.USE_POSTGRES:
            MfaSqlBase.metadata.drop_all(self.db_engine)
        elif self.db_path.exists():
            os.remove(self.db_path)

    def initialize_database(self) -> None:
        """
        Initialize the database with database schema
        """
        if self.database_initialized:
            return
        from montreal_forced_aligner.command_line.utils import check_databases

        if config.USE_POSTGRES:
            exist_check = True
            try:
                check_databases(self.identifier)
            except Exception:
                try:
                    subprocess.check_call(
                        [
                            "createdb",
                            f"--host={config.database_socket()}",
                            self.identifier,
                        ],
                        stderr=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                    )
                except Exception:
                    raise DatabaseError(
                        f"There was an error connecting to the {config.CURRENT_PROFILE_NAME} MFA database server "
                        f"at {config.database_socket()}. "
                        "Please ensure the server is initialized (mfa server init) or running (mfa server start)"
                    )
                exist_check = False
        else:
            exist_check = self.db_path.exists()
        self.database_initialized = True
        if config.CLEAN or getattr(self, "dirty", False):
            self.clean_working_directory()
        if exist_check:
            if config.CLEAN or getattr(self, "dirty", False):
                self.delete_database()
            else:
                return

        os.makedirs(self.output_directory, exist_ok=True)
        if config.USE_POSTGRES:
            with self.db_engine.connect() as conn:
                conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
                conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS pg_stat_statements"))
                conn.execute(sqlalchemy.text(f"select setseed({config.SEED / 32768})"))
                conn.commit()

        MfaSqlBase.metadata.create_all(self.db_engine)

    @property
    def db_engine(self) -> sqlalchemy.engine.Engine:
        """Database engine"""
        if self._db_engine is None:
            self._db_engine = self.construct_engine()
        return self._db_engine

    def get_next_primary_key(self, database_table):
        with self.session() as session:
            pk = session.query(sqlalchemy.func.max(database_table.id)).scalar()
            if not pk:
                pk = 0
        return pk + 1

    def create_new_current_workflow(self, workflow_type: WorkflowType, name: str = None):
        from montreal_forced_aligner.db import CorpusWorkflow

        with self.session() as session:
            if not name:
                name = workflow_type.name
            self._current_workflow = name

            session.query(CorpusWorkflow).update({"current": False})
            new_workflow = (
                session.query(CorpusWorkflow).filter(CorpusWorkflow.name == name).first()
            )
            if not new_workflow:
                new_workflow = CorpusWorkflow(
                    name=name,
                    workflow_type=workflow_type,
                    working_directory=os.path.join(self.output_directory, name),
                    current=True,
                )
                log_dir = os.path.join(new_workflow.working_directory, "log")
                os.makedirs(log_dir, exist_ok=True)
                session.add(new_workflow)
            else:
                new_workflow.current = True
            session.commit()

    def set_current_workflow(self, identifier):
        from montreal_forced_aligner.db import CorpusWorkflow

        with self.session() as session:
            session.query(CorpusWorkflow).update({CorpusWorkflow.current: False})
            wf = session.query(CorpusWorkflow).filter(CorpusWorkflow.name == identifier).first()
            wf.current = True
            self._current_workflow = identifier
            session.commit()

    @property
    def current_workflow(self) -> CorpusWorkflow:
        from montreal_forced_aligner.db import CorpusWorkflow

        with self.session() as session:
            wf = (
                session.query(CorpusWorkflow)
                .filter(CorpusWorkflow.current == True)  # noqa
                .first()
            )
        return wf

    @property
    def db_path(self) -> Path:
        """Connection path for sqlite database"""
        return self.output_directory.joinpath(f"{self.identifier}.db")

    @property
    def db_string(self) -> str:
        """Connection string for the database"""
        if config.USE_POSTGRES:
            return f"postgresql+psycopg2://@/{self.identifier}?host={config.database_socket()}"
        else:
            return f"sqlite:///{self.db_path}"

    def construct_engine(self, **kwargs) -> sqlalchemy.engine.Engine:
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
        db_string = self.db_string
        if not config.USE_POSTGRES:
            if kwargs.pop("read_only", False):
                db_string += "?mode=ro&nolock=1&uri=true"
        kwargs["pool_size"] = config.NUM_JOBS + 10
        kwargs["max_overflow"] = config.NUM_JOBS + 10
        e = sqlalchemy.create_engine(
            db_string,
            **kwargs,
        )

        return e

    @property
    def session(self) -> sqlalchemy.orm.scoped_session:
        """
        Construct database session

        Parameters
        ----------
        **kwargs
            Keyword arguments to pass to the Session

        Returns
        -------
        :class:`~sqlalchemy.orm.sessionmaker`
            SqlAlchemy session
        """
        if self._session is None:
            self._session = scoped_session(
                sessionmaker(bind=self.db_engine, expire_on_commit=False)
            )
        return self._session


class MfaWorker(metaclass=abc.ABCMeta):
    """
    Abstract class for MFA workers

    Attributes
    ----------
    dirty: bool
        Flag for whether an error was encountered in processing
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dirty = False

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
            "dirty": self.dirty,
        }

    @property
    @abc.abstractmethod
    def working_directory(self) -> Path:
        """Current working directory"""
        ...

    @property
    def working_log_directory(self) -> Path:
        """Current working log directory"""
        return self.working_directory.joinpath("log")

    @property
    @abc.abstractmethod
    def data_directory(self) -> Path:
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
        **kwargs,
    ):
        kwargs, skipped = type(self).extract_relevant_parameters(kwargs)
        super().__init__(**kwargs)
        self.initialized = False
        self.start_time = time.time()
        self.setup_logger()
        if skipped:
            logger.warning(f"Skipped the following configuration keys: {comma_join(skipped)}")

    def cleanup_logger(self):
        """Ensure that loggers are cleaned up on delete"""
        logger = logging.getLogger("mfa")
        handlers = logger.handlers[:]
        for handler in handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
            logger.removeHandler(handler)

    def setup(self) -> None:
        """Setup for worker"""
        self.check_previous_run()
        if hasattr(self, "initialize_database"):
            self.initialize_database()
        if hasattr(self, "inspect_database"):
            self.inspect_database()

    @property
    def working_directory(self) -> Path:
        """Alias for a folder that contains worker information, separate from the data directory"""
        return self.output_directory.joinpath(self._current_workflow)

    @classmethod
    def parse_args(
        cls, args: Optional[Dict[str, Any]], unknown_args: Optional[List[str]]
    ) -> MetaDict:
        """
        Class method for parsing configuration parameters from command line arguments

        Parameters
        ----------
        args: dict[str, Any]
            Parsed arguments
        unknown_args: list[str]
            Optional list of arguments that were not parsed

        Returns
        -------
        dict[str, Any]
            Dictionary of specified configuration parameters
        """
        from montreal_forced_aligner.data import Language

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
            if (name.endswith("_directory") and name != "audio_directory") or (
                name.endswith("_path")
                and name not in {"rules_path", "phone_groups_path", "topology_path"}
            ):
                continue
            if args is not None and name in args and args[name] is not None:
                if param_type == Language:
                    params[name] = param_type[args[name]]
                else:
                    params[name] = param_type(args[name])
            elif name in unknown_dict:
                params[name] = param_type(unknown_dict[name])
                if param_type == bool and not isinstance(unknown_dict[name], bool):
                    if unknown_dict[name].lower() == "false":
                        params[name] = False
        return params

    @classmethod
    def parse_parameters(
        cls,
        config_path: Optional[Path] = None,
        args: Optional[Dict[str, Any]] = None,
        unknown_args: Optional[typing.Iterable[str]] = None,
    ) -> MetaDict:
        """
        Parse configuration parameters from a config file and command line arguments

        Parameters
        ----------
        config_path: :class:`~pathlib.Path`, optional
            Path to yaml configuration file
        args: dict[str, Any]
            Parsed arguments
        unknown_args: list[str]
            Optional list of arguments that were not parsed

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
    def worker_config_path(self) -> str:
        """Path to worker's configuration in the working directory"""
        return os.path.join(self.output_directory, f"{self.data_source_identifier}.yaml")

    def cleanup(self) -> None:
        """
        Clean up loggers and output final message for top-level workers
        """
        try:
            if hasattr(self, "cleanup_connections"):
                self.cleanup_connections()
            if self.dirty:
                logger.error("There was an error in the run, please see the log.")
            else:
                logger.info(f"Done! Everything took {time.time() - self.start_time:.3f} seconds")
                if config.FINAL_CLEAN:
                    logger.debug(
                        "Cleaning up temporary files, use the --no_final_clean flag to keep temporary files."
                    )
                    if hasattr(self, "delete_database"):
                        if config.USE_POSTGRES:
                            proc = subprocess.run(
                                [
                                    "dropdb",
                                    f"--host={config.database_socket()}",
                                    "--if-exists",
                                    "--force",
                                    self.identifier,
                                ],
                                stderr=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                check=True,
                                encoding="utf-8",
                            )
                            logger.debug(f"Stdout: {proc.stdout}")
                            logger.debug(f"Stderr: {proc.stderr}")
                        else:
                            self.delete_database()
                    self.clean_working_directory()
            self.save_worker_config()
            self.cleanup_logger()
        except (NameError, ValueError):  # already cleaned up
            pass

    def save_worker_config(self) -> None:
        """Export worker configuration to its working directory"""
        if not os.path.exists(self.output_directory):
            return
        with mfa_open(self.worker_config_path, "w") as f:
            yaml.dump(self.configuration, f)

    def _validate_previous_configuration(self, conf: MetaDict) -> None:
        """
        Validate the current configuration against a previous configuration

        Parameters
        ----------
        conf: dict[str, Any]
            Previous run's configuration
        """
        from montreal_forced_aligner.utils import get_mfa_version

        self.dirty = False
        current_version = get_mfa_version()
        if not config.DEBUG and conf.get("version", current_version) != current_version:
            logger.debug(
                f"Previous run was on {conf['version']} version (new run: {current_version})"
            )
            self.dirty = True

    def check_previous_run(self) -> None:
        """
        Check whether a previous run has any conflicting settings with the current run.

        Returns
        -------
        bool
            Flag for whether the current run is compatible with the previous one
        """
        if not os.path.exists(self.worker_config_path):
            return True
        try:
            conf = load_configuration(self.worker_config_path)
            self._validate_previous_configuration(conf)
            if not config.CLEAN and self.dirty:
                logger.warning(
                    "The previous run had a different configuration than the current, which may cause issues."
                    " Please see the log for details or use --clean flag if issues are encountered."
                )
        except yaml.error.YAMLError:
            logger.warning("The previous run's configuration could not be loaded.")
            return False

    @property
    def identifier(self) -> str:
        """Combined identifier of the data source and workflow"""
        return self.data_source_identifier

    @property
    def output_directory(self) -> Path:
        """Root temporary directory to store all of this worker's files"""
        return config.TEMPORARY_DIRECTORY.joinpath(self.identifier)

    @property
    def log_file(self) -> Path:
        """Path to the worker's log file"""
        return self.output_directory.joinpath(f"{self.data_source_identifier}.log")

    def setup_logger(self) -> None:
        """
        Construct a logger for a command line run
        """
        from montreal_forced_aligner.helper import configure_logger
        from montreal_forced_aligner.utils import get_mfa_version

        current_version = get_mfa_version()
        # Remove previous directory if versions are different
        clean = False
        if os.path.exists(self.worker_config_path):
            conf = load_configuration(self.worker_config_path)
            if conf.get("version", current_version) != current_version:
                clean = True
        os.makedirs(self.output_directory, exist_ok=True)
        configure_logger("mfa", log_file=self.log_file)
        logger = logging.getLogger("mfa")
        if config.VERBOSE:
            try:
                response = requests.get(
                    "https://api.github.com/repos/MontrealCorpusTools/Montreal-Forced-Aligner/releases/latest"
                )
                latest_version = response.json()["tag_name"].replace("v", "")
                if current_version < latest_version:
                    logger.debug(
                        f"You are currently running an older version of MFA ({current_version}) than the latest available ({latest_version}). "
                        f"To update, please run mfa_update."
                    )
            except Exception:
                pass
        if re.search(r"\d+\.\d+\.\d+a", current_version) is not None:
            logger.debug(
                "Please be aware that you are running an alpha version of MFA. If you would like to install a more "
                "stable version, please visit https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html#installing-older-versions-of-mfa",
            )
        logger.debug(f"Beginning run for {self.data_source_identifier}")
        logger.debug(f'Using "{config.CURRENT_PROFILE_NAME}" profile')
        if config.USE_MP:
            logger.debug(f"Using multiprocessing with {config.NUM_JOBS}")
        else:
            logger.debug(f"NOT using multiprocessing with {config.NUM_JOBS}")
        logger.debug(f"Set up logger for MFA version: {current_version}")
        if clean or config.CLEAN:
            logger.debug("Cleaned previous run")


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
    def export_model(self, output_model_path: Path) -> None:
        """
        Abstract method to export an MFA model

        Parameters
        ----------
        output_model_path: :class:`~pathlib.Path`
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
    model_version: str
        Override for model version

    Attributes
    ----------
    iteration: int
        Current iteration
    """

    def __init__(self, num_iterations: int = 40, model_version: str = None, **kwargs):
        super().__init__(**kwargs)
        self.iteration: int = 0
        self.num_iterations = num_iterations
        self.model_version = model_version

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
    def pretrained_directory(cls) -> Path:
        """Directory that pretrained models are saved in"""
        from .config import get_temporary_directory

        path = get_temporary_directory().joinpath("pretrained_models", cls.model_type)
        path.mkdir(parents=True, exist_ok=True)
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
        if not cls.pretrained_directory().exists():
            return []
        available = []
        for f in cls.pretrained_directory().iterdir():
            if cls.valid_extension(f):
                available.append(f.stem)
        return available

    @classmethod
    def get_pretrained_path(cls, name: str, enforce_existence: bool = True) -> Path:
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
        Path
            Path to model
        """
        return cls.generate_path(cls.pretrained_directory(), name, enforce_existence)

    @classmethod
    @abc.abstractmethod
    def valid_extension(cls, filename: Path) -> bool:
        """Check whether a file has a valid extensions"""
        ...

    @classmethod
    @abc.abstractmethod
    def generate_path(
        cls, root: Path, name: str, enforce_existence: bool = True
    ) -> Optional[Path]:
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
        ...
