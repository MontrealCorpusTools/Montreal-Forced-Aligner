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

import sqlalchemy
import yaml

from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.db import CorpusWorkflow, MfaSqlBase
from montreal_forced_aligner.exceptions import KaldiProcessingError, MultiprocessingError
from montreal_forced_aligner.helper import comma_join, load_configuration, mfa_open

if TYPE_CHECKING:

    from montreal_forced_aligner.data import MfaArguments, WorkflowType

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
logger = logging.getLogger("mfa")


class KaldiFunction(metaclass=abc.ABCMeta):
    """
    Abstract class for running Kaldi functions
    """

    def __init__(self, args: MfaArguments):
        self.args = args
        self.db_string = self.args.db_string
        self.job_name = self.args.job_name
        self.log_path = self.args.log_path

    def run(self) -> typing.Generator:
        """Run the function, calls subclassed object's ``_run`` with error handling"""
        self.db_engine = sqlalchemy.create_engine(
            self.db_string,
            poolclass=sqlalchemy.NullPool,
            isolation_level="AUTOCOMMIT",
            logging_name=f"{type(self).__name__}_engine",
            pool_reset_on_return=None,
        ).execution_options(logging_token=f"{type(self).__name__}_engine")
        try:
            yield from self._run()
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
    def output_directory(self) -> str:
        """Root temporary directory"""
        ...

    def clean_working_directory(self) -> None:
        """Clean up previous runs"""
        shutil.rmtree(self.output_directory, ignore_errors=True)

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

    @property
    def model_output_directory(self) -> str:
        """Temporary directory containing all dictionary information"""
        return os.path.join(self.output_directory, "models")

    @dictionary_output_directory.setter
    def dictionary_output_directory(self, directory: str) -> None:
        self._dictionary_output_directory = directory

    @property
    def language_model_output_directory(self) -> str:
        """Temporary directory containing all dictionary information"""
        if self._language_model_output_directory:
            return self._language_model_output_directory
        return os.path.join(self.model_output_directory, "language_model")

    @language_model_output_directory.setter
    def language_model_output_directory(self, directory: str) -> None:
        self._language_model_output_directory = directory

    @property
    def acoustic_model_output_directory(self) -> str:
        """Temporary directory containing all dictionary information"""
        if self._acoustic_model_output_directory:
            return self._acoustic_model_output_directory
        return os.path.join(self.model_output_directory, "acoustic_model")

    @acoustic_model_output_directory.setter
    def acoustic_model_output_directory(self, directory: str) -> None:
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
        self.db_backend = GLOBAL_CONFIG.database_backend

        self._db_engine = None
        self._db_path = None
        self._session = None
        self.database_initialized = False

    def initialize_database(self) -> None:
        """
        Initialize the database with database schema
        """
        if self.database_initialized:
            return
        retcode = subprocess.call(
            [
                "createdb",
                f"--port={GLOBAL_CONFIG.current_profile.database_port}",
                self.identifier,
            ],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
        exist_check = retcode != 0
        with self.db_engine.connect() as conn:
            conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS pg_stat_statements"))
            conn.commit()
        self.database_initialized = True
        if exist_check:
            if GLOBAL_CONFIG.current_profile.clean:
                self.clean_working_directory()
                with self.db_engine.connect() as conn:
                    for tbl in reversed(MfaSqlBase.metadata.sorted_tables):
                        conn.execute(tbl.delete())
                    conn.commit()
            else:
                return

        os.makedirs(self.output_directory, exist_ok=True)

        MfaSqlBase.metadata.create_all(self.db_engine)

    @property
    def db_engine(self) -> sqlalchemy.engine.Engine:
        """Database engine"""
        if self._db_engine is None:
            self._db_engine = self.construct_engine()
        return self._db_engine

    def get_next_primary_key(self, database_table: MfaSqlBase):
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
    def db_string(self):
        """Connection string for the database"""
        return f"postgresql+psycopg2://localhost:{GLOBAL_CONFIG.current_profile.database_port}/{self.identifier}"

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
        e = sqlalchemy.create_engine(
            self.db_string,
            poolclass=sqlalchemy.NullPool,
            logging_name="main_process_engine",
            **kwargs,
        ).execution_options(logging_token="main_process_engine")

        return e

    @property
    def session(self, **kwargs) -> sqlalchemy.orm.sessionmaker:
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
            self._session = sqlalchemy.orm.sessionmaker(bind=self.db_engine, **kwargs)
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
        **kwargs,
    ):
        kwargs, skipped = type(self).extract_relevant_parameters(kwargs)
        super().__init__(**kwargs)
        self.initialized = False
        self.start_time = time.time()
        self.setup_logger()
        if skipped:
            logger.warning(f"Skipped the following configuration keys: {comma_join(skipped)}")

    def __del__(self):
        """Ensure that loggers are cleaned up on delete"""
        logger = logging.getLogger("mfa")
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)

    def setup(self) -> None:
        """Setup for worker"""
        self.check_previous_run()
        if hasattr(self, "initialize_database"):
            self.initialize_database()

    @property
    def working_directory(self) -> str:
        """Alias for a folder that contains worker information, separate from the data directory"""
        return os.path.join(self.output_directory, self._current_workflow)

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
                name.endswith("_path") and name not in {"rules_path", "groups_path"}
            ):
                continue
            if args is not None and name in args and args[name] is not None:
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
        config_path: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
        unknown_args: Optional[typing.Iterable[str]] = None,
    ) -> MetaDict:
        """
        Parse configuration parameters from a config file and command line arguments

        Parameters
        ----------
        config_path: str, optional
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
            if getattr(self, "_session", None) is not None:
                del self._session
                self._session = None

            if getattr(self, "_db_engine", None) is not None:
                del self._db_engine
                self._db_engine = None
            if self.dirty:
                logger.error("There was an error in the run, please see the log.")
            else:
                logger.info(f"Done! Everything took {time.time() - self.start_time:.3f} seconds")
            handlers = logger.handlers[:]
            for handler in handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.removeHandler(handler)
            self.save_worker_config()
        except (NameError, ValueError):  # already cleaned up
            pass

    def save_worker_config(self) -> None:
        """Export worker configuration to its working directory"""
        if not os.path.exists(self.output_directory):
            return
        with mfa_open(self.worker_config_path, "w") as f:
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
            logger.debug("Previous run ended in an error (maybe ctrl-c?)")
            clean = False
        if conf.get("version", current_version) != current_version:
            logger.debug(
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
                logger.debug(
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
            logger.warning(
                "The previous run had a different configuration than the current, which may cause issues."
                " Please see the log for details or use --clean flag if issues are encountered."
            )
        return clean

    @property
    def identifier(self) -> str:
        """Combined identifier of the data source and workflow"""
        return self.data_source_identifier

    @property
    def output_directory(self) -> str:
        """Root temporary directory to store all of this worker's files"""
        return os.path.join(GLOBAL_CONFIG.temporary_directory, self.identifier)

    @property
    def log_file(self) -> str:
        """Path to the worker's log file"""
        return os.path.join(self.output_directory, f"{self.data_source_identifier}.log")

    def setup_logger(self) -> None:
        """
        Construct a logger for a command line run
        """
        from montreal_forced_aligner.config import GLOBAL_CONFIG
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
        logger.debug(f"Beginning run for {self.data_source_identifier}")
        logger.debug(f'Using "{GLOBAL_CONFIG.current_profile_name}" profile')
        if GLOBAL_CONFIG.use_mp:
            logger.debug(f"Using multiprocessing with {GLOBAL_CONFIG.num_jobs}")
        else:
            logger.debug(f"NOT using multiprocessing with {GLOBAL_CONFIG.num_jobs}")
        logger.debug(f"Set up logger for MFA version: {current_version}")
        if clean or GLOBAL_CONFIG.clean:
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
        ...
