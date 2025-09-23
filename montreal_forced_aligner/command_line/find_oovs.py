"""Command line functions for validating corpora"""
from __future__ import annotations

from pathlib import Path

import rich_click as click

from montreal_forced_aligner import config
from montreal_forced_aligner.command_line.utils import (
    common_options,
    initialize_configuration,
    validate_corpus_directory,
    validate_dictionary,
    validate_output_directory,
)
from montreal_forced_aligner.validation.corpus_validator import TrainingValidator

__all__ = ["find_oovs_cli"]


@click.command(
    name="find_oovs",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Find all OOVs in a corpus",
)
@click.argument("corpus_directory", type=click.UNPROCESSED, callback=validate_corpus_directory)
@click.argument("dictionary_path", type=click.UNPROCESSED, callback=validate_dictionary)
@click.argument("output_directory", type=click.UNPROCESSED, callback=validate_output_directory)
@click.option(
    "--config_path",
    "-c",
    help="Path to config file to use for training.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--speaker_characters",
    "-s",
    help="Number of characters of file names to use for determining speaker, "
    "default is to use directory names.",
    type=str,
    default="0",
)
@click.option(
    "--audio_directory",
    "-a",
    help="Audio directory root to use for finding audio files.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def find_oovs_cli(context, **kwargs) -> None:
    """
    Check for OOVs in a corpus
    """
    config.FINAL_CLEAN = True
    initialize_configuration(context)

    output_directory = kwargs.get("output_directory", None)
    config_path = kwargs.get("config_path", None)
    corpus_directory = kwargs["corpus_directory"].absolute()
    dictionary_path = kwargs["dictionary_path"]
    validator = TrainingValidator(
        corpus_directory=corpus_directory,
        dictionary_path=dictionary_path,
        **TrainingValidator.parse_parameters(config_path, context.params, context.args),
    )
    try:
        validator.setup()
        validator.analyze_setup(output_directory=output_directory)
    except Exception:
        validator.dirty = True
        raise
    finally:
        validator.cleanup()
