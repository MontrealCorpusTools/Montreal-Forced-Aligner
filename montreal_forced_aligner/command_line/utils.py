"""Utility functions for command line commands"""
from __future__ import annotations

import functools
import logging
import os
import shutil
import subprocess
import typing
from pathlib import Path

import rich_click as click
import sqlalchemy
import yaml

from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.exceptions import (
    FileArgumentNotFoundError,
    ModelExtensionError,
    ModelTypeNotSupportedError,
    NoDefaultSpeakerDictionaryError,
    PretrainedModelNotFoundError,
)
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.models import MODEL_TYPES

__all__ = [
    "validate_acoustic_model",
    "validate_g2p_model",
    "validate_ivector_extractor",
    "validate_language_model",
    "validate_dictionary",
    "check_databases",
]


def common_options(f: typing.Callable) -> typing.Callable:
    """
    Add common MFA cli options to a given command
    """
    options = [
        click.option(
            "-p",
            "--profile",
            help='Configuration profile to use, defaults to "global"',
            type=str,
            default=None,
        ),
        click.option(
            "--temporary_directory",
            "-t",
            "temporary_directory",
            help=f"Set the default temporary directory, default is {GLOBAL_CONFIG.temporary_directory}",
            type=str,
            default=GLOBAL_CONFIG.temporary_directory,
        ),
        click.option(
            "-j",
            "--num_jobs",
            "num_jobs",
            help=f"Set the number of processes to use by default, defaults to {GLOBAL_CONFIG.num_jobs}",
            type=int,
            default=None,
        ),
        click.option(
            "--clean/--no_clean",
            "clean",
            help=f"Remove files from previous runs, default is {GLOBAL_CONFIG.clean}",
            default=None,
        ),
        click.option(
            "--verbose/--no_verbose",
            "-v/-nv",
            "verbose",
            help=f"Output debug messages, default is {GLOBAL_CONFIG.verbose}",
            default=None,
        ),
        click.option(
            "--quiet/--no_quiet",
            "-q/-nq",
            "quiet",
            help=f"Suppress all output messages (overrides verbose), default is {GLOBAL_CONFIG.quiet}",
            default=None,
        ),
        click.option(
            "--overwrite/--no_overwrite",
            "overwrite",
            help=f"Overwrite output files when they exist, default is {GLOBAL_CONFIG.overwrite}",
            default=None,
        ),
        click.option(
            "--use_mp/--no_use_mp",
            "use_mp",
            help="Turn on/off multiprocessing. Multiprocessing is recommended will allow for faster executions.",
            default=None,
        ),
        click.option(
            "--debug/--no_debug",
            "-d/-nd",
            "debug",
            help=f"Run extra steps for debugging issues, default is {GLOBAL_CONFIG.debug}",
            default=None,
        ),
        click.option(
            "--single_speaker",
            "single_speaker",
            is_flag=True,
            help="Single speaker mode creates multiprocessing splits based on utterances rather than speakers.",
            default=False,
        ),
        click.option(
            "--textgrid_cleanup/--no_textgrid_cleanup",
            "cleanup_textgrids",
            help="Turn on/off post-processing of TextGrids that cleans up "
            "silences and recombines compound words and clitics.",
            default=None,
        ),
    ]
    options.reverse()
    return functools.reduce(lambda x, opt: opt(x), options, f)


def validate_model_arg(name: str, model_type: str) -> Path:
    """
    Validate pretrained model name argument

    Parameters
    ----------
    name: str
        Name of model
    model_type: str
        Type of model

    Returns
    -------
    path_like
        Full path of validated model

    Raises
    ------
    :class:`~montreal_forced_aligner.exceptions.ModelTypeNotSupportedError`
        If the type of model is not supported
    :class:`~montreal_forced_aligner.exceptions.FileArgumentNotFoundError`
        If the file specified is not found
    :class:`~montreal_forced_aligner.exceptions.PretrainedModelNotFoundError`
        If the pretrained model specified is not found
    :class:`~montreal_forced_aligner.exceptions.ModelExtensionError`
        If the extension is not valid for the specified model type
    :class:`~montreal_forced_aligner.exceptions.NoDefaultSpeakerDictionaryError`
        If a multispeaker dictionary does not have a default dictionary
    """
    if model_type not in MODEL_TYPES:
        raise click.BadParameter(str(ModelTypeNotSupportedError(model_type, MODEL_TYPES)))
    model_class = MODEL_TYPES[model_type]
    available_models = model_class.get_available_models()
    model_class = MODEL_TYPES[model_type]
    if name in available_models:
        name = model_class.get_pretrained_path(name)
    else:
        if isinstance(name, str):
            name = Path(name)
    if model_class.valid_extension(name):
        if not name.exists():
            raise click.BadParameter(str(FileArgumentNotFoundError(name)))
        if model_type == "dictionary" and name.suffix.lower() == ".yaml":
            with mfa_open(name, "r") as f:
                data = yaml.load(f, Loader=yaml.Loader)
                paths = sorted(set(data.values()))
                for path in paths:
                    validate_model_arg(path, "dictionary")
                if "default" not in data:
                    raise click.BadParameter(str(NoDefaultSpeakerDictionaryError()))
    else:
        if name.exists():
            if name.suffix:
                raise click.BadParameter(
                    str(ModelExtensionError(name, model_type, model_class.extensions))
                )
        else:
            raise click.BadParameter(
                str(PretrainedModelNotFoundError(name, model_type, available_models))
            )
    return name


def validate_acoustic_model(ctx, param, value):
    """Validation callback for acoustic model paths"""
    if value:
        return validate_model_arg(value, "acoustic")


def validate_dictionary(ctx, param, value):
    """Validation callback for dictionary paths"""
    if value:
        return validate_model_arg(value, "dictionary")


def validate_language_model(ctx, param, value):
    """Validation callback for language model paths"""
    if value:
        return validate_model_arg(value, "language_model")


def validate_g2p_model(ctx, param, value):
    """Validation callback for G2P model paths"""
    return validate_model_arg(value, "g2p")


def validate_tokenizer_model(ctx, param, value):
    """Validation callback for tokenizer model paths"""
    return validate_model_arg(value, "tokenizer")


def validate_ivector_extractor(ctx, param, value):
    """Validation callback for ivector extractor paths"""
    if value == "speechbrain":
        return value
    return validate_model_arg(value, "ivector")


def cleanup_logger():
    logger = logging.getLogger("mfa")
    handlers = logger.handlers[:]
    for handler in handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)


def configure_pg(directory):
    configuration_updates = {
        "#log_min_duration_statement = -1": "log_min_duration_statement = 5000",
        "#enable_partitionwise_join = off": "enable_partitionwise_join = on",
        "#enable_partitionwise_aggregate = off": "enable_partitionwise_aggregate = on",
        "#unix_socket_directories = ''": f"unix_socket_directories = '{GLOBAL_CONFIG.database_socket}'",
        "#listen_addresses = 'localhost'": "listen_addresses = ''",
    }
    if not GLOBAL_CONFIG.current_profile.database_limited_mode:
        configuration_updates.update(
            {
                "#maintenance_work_mem = 64MB": "maintenance_work_mem = 500MB",
                "#work_mem = 4MB": "work_mem = 128MB",
                "shared_buffers = 128MB": "shared_buffers = 256MB",
                "max_connections = 100": "max_connections = 10000",
            }
        )
    else:
        configuration_updates.update(
            {
                "#wal_level = replica": "wal_level = minimal",
                "#fsync = on": "fsync = off",
                "#synchronous_commit = on": "synchronous_commit = off",
                "#full_page_writes = on": "full_page_writes = off",
                "#max_wal_senders = 10": "max_wal_senders = 0",
            }
        )
    with mfa_open(os.path.join(directory, "postgresql.conf"), "r") as f:
        config = f.read()
    for query, rep in configuration_updates.items():
        config = config.replace(query, rep)
    with mfa_open(os.path.join(directory, "postgresql.conf"), "w") as f:
        f.write(config)


def check_databases(db_name=None) -> None:
    """Check for existence of necessary databases"""
    logger = logging.getLogger("mfa")
    GLOBAL_CONFIG.load()
    logger.debug(f"Checking the {GLOBAL_CONFIG.current_profile_name} MFA database server...")

    try:
        engine = sqlalchemy.create_engine(
            f"postgresql+psycopg2://@/{db_name}?host={GLOBAL_CONFIG.database_socket}",
            poolclass=sqlalchemy.NullPool,
            pool_reset_on_return=None,
            logging_name="check_databases_engine",
            isolation_level="AUTOCOMMIT",
        ).execution_options(logging_token="check_databases_engine")
        with engine.connect():
            pass
        logger.debug(f"Connected to {GLOBAL_CONFIG.current_profile_name} MFA database server!")
    except Exception:
        raise
    finally:
        logger.error(
            f"There was an error connecting to the {GLOBAL_CONFIG.current_profile_name} MFA database server."
        )
        logger.error(
            "Please ensure the server is initialized (mfa server init) or running (mfa server start)"
        )


def initialize_server() -> None:
    """Initialize the MFA server for the current profile"""
    GLOBAL_CONFIG.load()
    logger = logging.getLogger("mfa")
    logger.info(f"Initializing the {GLOBAL_CONFIG.current_profile_name} MFA database server...")

    db_directory = os.path.join(
        GLOBAL_CONFIG["temporary_directory"], f"pg_mfa_{GLOBAL_CONFIG.current_profile_name}"
    )
    init_log_path = os.path.join(
        GLOBAL_CONFIG["temporary_directory"],
        f"pg_init_log_{GLOBAL_CONFIG.current_profile_name}.txt",
    )
    os.makedirs(GLOBAL_CONFIG["temporary_directory"], exist_ok=True)
    with open(init_log_path, "w") as log_file:
        try:
            subprocess.check_call(
                ["initdb", "-D", db_directory, "--encoding=UTF8"],
                stdout=log_file,
                stderr=log_file,
            )
            configure_pg(db_directory)
        except Exception:
            logger.error(
                f"There was an issue initializing the server, please refer to {init_log_path} for more details"
            )
            raise
        start_server()
        try:
            subprocess.check_call(
                [
                    "createuser",
                    "-h",
                    GLOBAL_CONFIG.database_socket,
                    "-s",
                    "postgres",
                ],
                stdout=log_file,
                stderr=log_file,
            )
        except Exception:
            logger.error(
                f"There was an issue initializing the server, please refer to {init_log_path} for more details"
            )
            raise


def start_server() -> None:
    """Start the MFA server for the current profile"""
    GLOBAL_CONFIG.load()
    logger = logging.getLogger("mfa")
    logger.info(f"Starting the {GLOBAL_CONFIG.current_profile_name} MFA database server...")

    db_directory = os.path.join(
        GLOBAL_CONFIG["temporary_directory"], f"pg_mfa_{GLOBAL_CONFIG.current_profile_name}"
    )
    log_path = os.path.join(
        GLOBAL_CONFIG["temporary_directory"], f"pg_log_{GLOBAL_CONFIG.current_profile_name}.txt"
    )
    os.makedirs(GLOBAL_CONFIG["temporary_directory"], exist_ok=True)
    subprocess.check_call(
        [
            "pg_ctl",
            "-D",
            db_directory,
            "-l",
            log_path,
            "start",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    logger.info(f"{GLOBAL_CONFIG.current_profile_name} MFA database server started!")


def stop_server(mode: str = "fast") -> None:
    """Stop the MFA server for the current profile"""
    logger = logging.getLogger("mfa")
    logger.info(f"Stopping the {GLOBAL_CONFIG.current_profile_name} MFA database server...")
    GLOBAL_CONFIG.load()
    sqlalchemy.orm.session.close_all_sessions()

    db_directory = os.path.join(
        GLOBAL_CONFIG["temporary_directory"], f"pg_mfa_{GLOBAL_CONFIG.current_profile_name}"
    )
    try:
        subprocess.check_call(
            ["pg_ctl", "-D", db_directory, "-m", mode, "stop"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        pass


def remove_server() -> None:
    """Remove the MFA server for the current profile"""
    stop_server(mode="immediate")
    logger = logging.getLogger("mfa")
    logger.info(f"Deleting the {GLOBAL_CONFIG.current_profile_name} MFA database server...")
    GLOBAL_CONFIG.load()

    db_directory = GLOBAL_CONFIG.current_profile.temporary_directory.joinpath(
        f"pg_mfa_{GLOBAL_CONFIG.current_profile_name}"
    )
    if db_directory.exists():
        shutil.rmtree(db_directory)
