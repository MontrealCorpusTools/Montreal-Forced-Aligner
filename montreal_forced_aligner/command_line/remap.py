"""Command line functions for adapting acoustic models to new data"""
from __future__ import annotations

from pathlib import Path

import rich_click as click

from montreal_forced_aligner.command_line.utils import (
    common_options,
    initialize_configuration,
    validate_acoustic_model,
    validate_corpus_directory,
    validate_dictionary,
)
from montreal_forced_aligner.corpus.remapper import AlignmentRemapper
from montreal_forced_aligner.dictionary.remapper import DictionaryRemapper

__all__ = ["remap_cli", "remap_alignments_cli", "remap_dictionary_cli"]


@click.group(name="remap", short_help="Remap files to a new phone set")
@click.help_option("-h", "--help")
def remap_cli() -> None:
    """
    Remap dictionary and alignment files to a new phone set
    """
    pass


@remap_cli.command(
    name="alignments",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Remap alignments to a new phone set",
)
@click.argument("corpus_directory", type=click.UNPROCESSED, callback=validate_corpus_directory)
@click.argument(
    "phone_mapping_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "output_directory", type=click.Path(file_okay=False, dir_okay=True, path_type=Path)
)
@click.option(
    "--output_format",
    help="Format for aligned output files (default is long_textgrid).",
    default="long_textgrid",
    type=click.Choice(["long_textgrid", "short_textgrid", "json", "csv"]),
)
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def remap_alignments_cli(context: click.Context, **kwargs) -> None:
    """
    Adapt an acoustic model to a new corpus.
    """
    initialize_configuration(context)
    config_path = kwargs.get("config_path", None)
    corpus_directory = kwargs["corpus_directory"]
    phone_mapping_path = kwargs["phone_mapping_path"]
    output_directory = kwargs["output_directory"]
    output_format = kwargs["output_format"]
    remapper = AlignmentRemapper(
        corpus_directory=corpus_directory,
        phone_mapping_path=phone_mapping_path,
        **AlignmentRemapper.parse_parameters(config_path, context.params, context.args),
    )

    try:
        remapper.setup()
        remapper.remap_alignments(output_directory, output_format)
    except Exception:
        remapper.dirty = True
        raise
    finally:
        remapper.cleanup()


@remap_cli.command(
    name="dictionary",
    context_settings=dict(
        ignore_unknown_options=False,
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
def remap_dictionary_cli(context: click.Context, **kwargs) -> None:
    """
    Adapt an acoustic model to a new corpus.
    """
    initialize_configuration(context)
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
