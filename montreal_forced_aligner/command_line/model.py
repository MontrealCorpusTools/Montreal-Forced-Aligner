"""Command line functions for interacting with MFA models"""
from __future__ import annotations

import logging
import os
import shutil
import typing

import click

from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.data import PhoneSetType
from montreal_forced_aligner.exceptions import (
    ModelLoadError,
    ModelSaveError,
    ModelTypeNotSupportedError,
    MultipleModelTypesFoundError,
    PretrainedModelNotFoundError,
)
from montreal_forced_aligner.models import MODEL_TYPES, Archive, ModelManager, guess_model_type

__all__ = [
    "model_cli",
    "save_model_cli",
    "download_model_cli",
    "list_model_cli",
    "inspect_model_cli",
]


@click.group(name="model", short_help="Download, inspect, and save models")
@click.help_option("-h", "--help")
def model_cli() -> None:
    """
    Inspect, download, and save pretrained MFA models
    """
    pass


@model_cli.command(name="download", short_help="Download pretrained models")
@click.argument("model_type", type=click.Choice(sorted(MODEL_TYPES)))
@click.argument("model_name", nargs=-1, type=str)
@click.option(
    "--github_token",
    help="Personal access token to use for requests to GitHub to increase rate limit.",
    type=str,
    default=None,
)
@click.option(
    "--ignore_cache",
    is_flag=True,
    help="Flag to ignore existing downloaded models and force a re-download.",
    default=False,
)
@click.help_option("-h", "--help")
def download_model_cli(
    model_type: str, model_name: typing.List[str], github_token: str, ignore_cache: bool
) -> None:
    """
    Download pretrained models from the MFA repository. If no model names are specified, the list of all downloadable models
    of the given model type will be printed.
    """
    manager = ModelManager(token=github_token)
    if model_name:
        for name in model_name:
            manager.download_model(model_type, name, ignore_cache)
    else:
        manager.print_remote_models(model_type)


@model_cli.command(name="list", short_help="List available models")
@click.argument("model_type", type=click.Choice(sorted(MODEL_TYPES)))
@click.help_option("-h", "--help")
def list_model_cli(model_type: str) -> None:
    """
    List of locally saved models.
    """
    manager = ModelManager(token=GLOBAL_CONFIG.github_token)
    manager.print_local_models(model_type)


@model_cli.command(name="inspect", short_help="Inspect a model")
@click.argument("model_type", type=click.Choice(sorted(MODEL_TYPES)))
@click.argument("model", type=str)
@click.help_option("-h", "--help")
def inspect_model_cli(model_type: str, model: str) -> None:
    """
    Inspect a model and print out its metadata.
    """
    from montreal_forced_aligner.config import GLOBAL_CONFIG, get_temporary_directory

    GLOBAL_CONFIG.current_profile.clean = True
    GLOBAL_CONFIG.current_profile.temporary_directory = os.path.join(
        get_temporary_directory(), "model_inspect"
    )
    shutil.rmtree(GLOBAL_CONFIG.current_profile.temporary_directory, ignore_errors=True)
    if model_type and model_type not in MODEL_TYPES:
        raise ModelTypeNotSupportedError(model_type, MODEL_TYPES)
    elif model_type:
        model_type = model_type.lower()
    possible_model_types = guess_model_type(model)
    if not possible_model_types:
        if model_type:
            model_class = MODEL_TYPES[model_type]
            path = model_class.get_pretrained_path(model)
            if path is None:
                raise PretrainedModelNotFoundError(
                    model, model_type, model_class.get_available_models()
                )
        else:
            found_model_types = []
            path = None
            for model_type, model_class in MODEL_TYPES.items():
                p = model_class.get_pretrained_path(model)
                if p is not None:
                    path = p
                    found_model_types.append(model_type)
            if len(found_model_types) > 1:
                raise MultipleModelTypesFoundError(model, found_model_types)
            if path is None:
                raise PretrainedModelNotFoundError(model)
        model = path
    working_dir = os.path.join(get_temporary_directory(), "models", "inspect")
    ext = os.path.splitext(model)[1]
    if model_type:
        if model_type == MODEL_TYPES["dictionary"]:
            m = MODEL_TYPES[model_type](model, working_dir, phone_set_type=PhoneSetType.AUTO)
        else:
            m = MODEL_TYPES[model_type](model, working_dir)
    else:
        m = None
        if ext == Archive.extensions[0]:  # Figure out what kind of model it is
            a = Archive(model, working_dir)
            m = a.get_subclass_object()
        if not m:
            raise ModelLoadError(path)
    m.pretty_print()


@model_cli.command(name="save", short_help="Save a model")
@click.argument("model_type", type=click.Choice(sorted(MODEL_TYPES)))
@click.argument("path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option(
    "--name", help="Name to use as reference (defaults to the name of the zip file).", type=str
)
@click.option(
    "--overwrite/--no_overwrite",
    "overwrite",
    help=f"Overwrite output files when they exist, default is {GLOBAL_CONFIG.overwrite}",
    default=GLOBAL_CONFIG.overwrite,
)
@click.help_option("-h", "--help")
def save_model_cli(path: str, model_type: str, name: str, overwrite: bool) -> None:
    """
    Save a model to pretrained folder for later use

    Parameters
    ----------
    path: str
        Path to model
    model_type: str
        Type of model
    """
    logger = logging.getLogger("mfa")
    model_name = os.path.splitext(os.path.basename(path))[0]
    model_class = MODEL_TYPES[model_type]
    if name:
        out_path = model_class.get_pretrained_path(name, enforce_existence=False)
    else:
        out_path = model_class.get_pretrained_path(model_name, enforce_existence=False)
    if not overwrite and os.path.exists(out_path):
        raise ModelSaveError(out_path)
    shutil.copyfile(path, out_path)
    logger.info(
        f"Saved model to {name}, you can now use {name} in place of paths in mfa commands."
    )
