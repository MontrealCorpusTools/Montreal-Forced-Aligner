"""Command line functions for adapting acoustic models to new data"""
from __future__ import annotations

import os
from pathlib import Path

import rich_click as click

from montreal_forced_aligner import config
from montreal_forced_aligner.command_line.utils import (
    common_options,
    validate_acoustic_model,
    validate_dictionary,
)
from montreal_forced_aligner.dictionary.remapper import DictionaryRemapper

__all__ = ["remap_dictionary_cli"]


@click.command(
    name="remap_dictionary",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Remap a dictionary to a new phone set for use with an acoustic model",
)
@click.argument("dictionary_path", type=click.UNPROCESSED, callback=validate_dictionary)
@click.argument("acoustic_model_path", type=click.UNPROCESSED, callback=validate_acoustic_model)
@click.argument(
    "phone_mapping_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "output_dictionary_path", type=click.Path(file_okay=True, dir_okay=False, path_type=Path)
)
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def remap_dictionary_cli(context, **kwargs) -> None:
    """
    Adapt an acoustic model to a new corpus.
    """
    if kwargs.get("profile", None) is not None:
        os.environ[config.MFA_PROFILE_VARIABLE] = kwargs.pop("profile")
    config.update_configuration(kwargs)
    config_path = kwargs.get("config_path", None)
    dictionary_path = kwargs["dictionary_path"]
    acoustic_model_path = kwargs["acoustic_model_path"]
    phone_mapping_path = kwargs["phone_mapping_path"]
    output_dictionary_path = kwargs["output_dictionary_path"]
    remapper = DictionaryRemapper(
        dictionary_path=dictionary_path,
        acoustic_model_path=acoustic_model_path,
        phone_mapping_path=phone_mapping_path,
        **DictionaryRemapper.parse_parameters(config_path, context.params, context.args),
    )

    try:
        remapper.setup()
        remapper.remap(output_dictionary_path)
    except Exception:
        remapper.dirty = True
        raise
    finally:
        remapper.cleanup()
