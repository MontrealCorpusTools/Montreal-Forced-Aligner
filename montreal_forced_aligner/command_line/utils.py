"""Utility functions for command line commands"""
from __future__ import annotations

import functools
import os
import subprocess
import typing

import click
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
            default="global",
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
            "--database_backend",
            default=GLOBAL_CONFIG.database_backend,
            help="Backend database to use for storing metadata. "
            f"Currently set to {GLOBAL_CONFIG.database_backend}.",
            type=click.Choice(["sqlite", "psycopg2"]),
        ),
        click.option(
            "-j",
            "--num_jobs",
            "num_jobs",
            help=f"Set the number of processes to use by default, defaults to {GLOBAL_CONFIG.num_jobs}",
            type=int,
            default=GLOBAL_CONFIG.num_jobs,
        ),
        click.option(
            "--clean/--no_clean",
            "clean",
            help=f"Remove files from previous runs, default is {GLOBAL_CONFIG.clean}",
            default=GLOBAL_CONFIG.clean,
        ),
        click.option(
            "--verbose/--no_verbose",
            "-v/-nv",
            "verbose",
            help=f"Output debug messages, default is {GLOBAL_CONFIG.verbose}",
            default=GLOBAL_CONFIG.verbose,
        ),
        click.option(
            "--quiet/--no_quiet",
            "-q/-nq",
            "quiet",
            help=f"Suppress all output messages (overrides verbose), default is {GLOBAL_CONFIG.quiet}",
            default=GLOBAL_CONFIG.quiet,
        ),
        click.option(
            "--overwrite/--no_overwrite",
            "overwrite",
            help=f"Overwrite output files when they exist, default is {GLOBAL_CONFIG.overwrite}",
            default=GLOBAL_CONFIG.overwrite,
        ),
        click.option(
            "--use_mp/--no_use_mp",
            "use_mp",
            help="Turn on/off multiprocessing. Multiprocessing is recommended will allow for faster executions.",
            default=GLOBAL_CONFIG.use_mp,
        ),
        click.option(
            "--debug/--no_debug",
            "-d/-nd",
            "debug",
            help=f"Run extra steps for debugging issues, default is {GLOBAL_CONFIG.debug}",
            default=GLOBAL_CONFIG.debug,
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
            default=GLOBAL_CONFIG.cleanup_textgrids,
        ),
    ]
    options.reverse()
    return functools.reduce(lambda x, opt: opt(x), options, f)


def validate_model_arg(name: str, model_type: str) -> str:
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
    str
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
    elif model_class.valid_extension(name):
        if not os.path.exists(name):
            raise click.BadParameter(str(FileArgumentNotFoundError(name)))
        if model_type == "dictionary" and os.path.splitext(name)[1].lower() == ".yaml":
            with mfa_open(name, "r") as f:
                data = yaml.safe_load(f)
                found_default = False
                for speaker, path in data.items():
                    if speaker == "default":
                        found_default = True
                    path = validate_model_arg(path, "dictionary")
                if not found_default:
                    raise click.BadParameter(str(NoDefaultSpeakerDictionaryError()))
    else:
        if os.path.exists(name):
            if os.path.splitext(name)[1]:
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
    """Validation callback for G2O model paths"""
    return validate_model_arg(value, "g2p")


def validate_ivector_extractor(ctx, param, value):
    """Validation callback for ivector extractor paths"""
    return validate_model_arg(value, "ivector")


def check_databases() -> None:
    """Check for existence of necessary databases"""
    from montreal_forced_aligner.abc import DatabaseBackend
    from montreal_forced_aligner.config import GLOBAL_CONFIG

    if GLOBAL_CONFIG["database_backend"] == DatabaseBackend.POSTGRES.value:
        db_directory = os.path.join(GLOBAL_CONFIG["temporary_directory"], "pg_mfa")
        init_log_path = os.path.join(GLOBAL_CONFIG["temporary_directory"], "pg_init_log.txt")
        log_path = os.path.join(GLOBAL_CONFIG["temporary_directory"], "pg_log.txt")
        os.makedirs(GLOBAL_CONFIG["temporary_directory"], exist_ok=True)
        create = not os.path.exists(db_directory)
        with open(init_log_path, "w") as log_file:
            if create:
                subprocess.check_call(
                    ["initdb", "-D", db_directory, "--encoding=UTF8"],
                    stdout=log_file,
                    stderr=log_file,
                )
            try:
                e = sqlalchemy.create_engine(
                    f"postgresql+psycopg2://{os.getlogin()}@localhost/pg_mfa"
                )
                with e.connect():
                    pass
            except sqlalchemy.exc.OperationalError:
                subprocess.check_call(
                    ["pg_ctl", "-D", db_directory, "-l", log_path, "restart"],
                    stdout=log_file,
                    stderr=log_file,
                )
