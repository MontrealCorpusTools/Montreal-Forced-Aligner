"""Command line functions for interacting with MFA models"""
from __future__ import annotations

import logging
import shutil
import typing
from pathlib import Path

import rich_click as click

from montreal_forced_aligner import config
from montreal_forced_aligner.command_line.utils import common_options, validate_dictionary
from montreal_forced_aligner.data import PhoneSetType
from montreal_forced_aligner.dictionary.multispeaker import MultispeakerDictionary
from montreal_forced_aligner.exceptions import (
    ModelLoadError,
    ModelSaveError,
    ModelTypeNotSupportedError,
    MultipleModelTypesFoundError,
    PhoneMismatchError,
    PretrainedModelNotFoundError,
)
from montreal_forced_aligner.models import MODEL_TYPES, Archive, ModelManager, guess_model_type

__all__ = [
    "model_cli",
    "save_model_cli",
    "download_model_cli",
    "list_model_cli",
    "inspect_model_cli",
    "add_words_cli",
]


@click.group(name="model", short_help="Download, inspect, and save models")
@click.help_option("-h", "--help")
def model_cli() -> None:
    """
    Inspect, download, and save pretrained MFA models and dictionaries
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
    "--version",
    help="Specific version of model to download rather than the latest.",
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
    model_type: str,
    model_name: typing.List[str],
    github_token: str,
    version: str,
    ignore_cache: bool,
) -> None:
    """
    Download pretrained models from the MFA repository. If no model names are specified, the list of all downloadable models
    of the given model type will be printed.
    """
    manager = ModelManager(token=github_token, ignore_cache=ignore_cache)
    if model_name:
        for name in model_name:
            manager.download_model(model_type, name, version=version)
    else:
        manager.print_remote_models(model_type)


@model_cli.command(name="list", short_help="List available models")
@click.argument("model_type", type=click.Choice(sorted(MODEL_TYPES)))
@click.help_option("-h", "--help")
def list_model_cli(model_type: str) -> None:
    """
    List of locally saved models.
    """
    manager = ModelManager(token=config.GITHUB_TOKEN)
    manager.print_local_models(model_type)


@model_cli.command(name="inspect", short_help="Inspect a model")
@click.argument("model_type", type=click.Choice(sorted(MODEL_TYPES)))
@click.argument("model", type=str)
@click.help_option("-h", "--help")
def inspect_model_cli(model_type: str, model: str) -> None:
    """
    Inspect a model and print out its metadata.
    """

    config.CLEAN = True
    config.TEMPORARY_DIRECTORY = config.get_temporary_directory().joinpath("model_inspect")
    shutil.rmtree(config.TEMPORARY_DIRECTORY, ignore_errors=True)
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
    working_dir = config.get_temporary_directory().joinpath("models", "inspect")
    if isinstance(model, str):
        model = Path(model)
    ext = model.suffix
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


@model_cli.command(name="add_words", short_help="Add words to a dictionary")
@click.argument("dictionary_path", type=click.UNPROCESSED, callback=validate_dictionary)
@click.argument("new_pronunciations_path", type=click.UNPROCESSED, callback=validate_dictionary)
@click.help_option("-h", "--help")
@common_options
@click.pass_context
def add_words_cli(context, **kwargs) -> None:
    """
    Add words from one pronunciation dictionary to another pronunciation dictionary,
    so long as the new pronunciations do not contain any new phones
    """
    if kwargs.get("profile", None) is not None:
        config.profile = kwargs.pop("profile")
    config.update_configuration(kwargs)

    config.CLEAN = True
    dictionary_path = kwargs.get("dictionary_path", None)
    new_pronunciations_path = kwargs.get("new_pronunciations_path", None)
    base_dictionary = MultispeakerDictionary(dictionary_path=dictionary_path)
    base_dictionary.dictionary_setup()
    new_pronunciations = MultispeakerDictionary(dictionary_path=new_pronunciations_path)
    new_pronunciations.dictionary_setup()
    new_phones = set()
    for phone in new_pronunciations.non_silence_phones:
        if phone not in base_dictionary.non_silence_phones:
            new_phones.add(phone)
    if new_phones:
        raise PhoneMismatchError(new_phones)

    new_words = new_pronunciations.words_for_export(probability=True)
    base_dictionary.add_words(new_words)
    base_dictionary.export_lexicon(
        base_dictionary._default_dictionary_id,
        base_dictionary.dictionary_model.path,
        probability=True,
    )
    new_pronunciations.cleanup_connections()
    base_dictionary.cleanup_connections()


@model_cli.command(name="save", short_help="Save a model")
@click.argument("model_type", type=click.Choice(sorted(MODEL_TYPES)))
@click.argument(
    "path", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--name", help="Name to use as reference (defaults to the name of the zip file).", type=str
)
@click.option(
    "--overwrite/--no_overwrite",
    "overwrite",
    help=f"Overwrite output files when they exist, default is {config.OVERWRITE}",
    default=config.OVERWRITE,
)
@click.help_option("-h", "--help")
def save_model_cli(path: Path, model_type: str, name: str, overwrite: bool) -> None:
    """
    Save a model to pretrained folder for later use

    Parameters
    ----------
    path: :class:`~pathlib.Path`
        Path to model
    model_type: str
        Type of model
    """
    logger = logging.getLogger("mfa")
    model_name = path.stem
    model_class = MODEL_TYPES[model_type]
    if name:
        out_path = model_class.get_pretrained_path(name, enforce_existence=False)
    else:
        out_path = model_class.get_pretrained_path(model_name, enforce_existence=False)
    if not overwrite and out_path.exists():
        raise ModelSaveError(out_path)
    shutil.copyfile(path, out_path)
    logger.info(
        f"Saved model to {name}, you can now use {name} in place of paths in mfa commands."
    )
