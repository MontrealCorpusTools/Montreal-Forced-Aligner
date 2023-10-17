"""Command line functions for training dictionaries with pronunciation probabilities"""
from __future__ import annotations

from pathlib import Path

import rich_click as click

from montreal_forced_aligner import config
from montreal_forced_aligner.alignment.pretrained import DictionaryTrainer
from montreal_forced_aligner.command_line.utils import (
    common_options,
    validate_acoustic_model,
    validate_dictionary,
)

__all__ = ["train_dictionary_cli"]


@click.command(
    name="train_dictionary",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Calculate pronunciation probabilities",
)
@click.argument(
    "corpus_directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument("dictionary_path", type=click.UNPROCESSED, callback=validate_dictionary)
@click.argument("acoustic_model_path", type=click.UNPROCESSED, callback=validate_acoustic_model)
@click.argument(
    "output_directory", type=click.Path(file_okay=False, dir_okay=True, path_type=Path)
)
@click.option(
    "--config_path",
    "-c",
    help="Path to config file to use for training.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--silence_probabilities",
    is_flag=True,
    help="Flag for saving silence information for pronunciations.",
    default=False,
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
def train_dictionary_cli(context, **kwargs) -> None:
    """
    Calculate pronunciation probabilities for a dictionary based on alignment results in a corpus.
    """
    if kwargs.get("profile", None) is not None:
        config.profile = kwargs.pop("profile")
    config.update_configuration(kwargs)
    config_path = kwargs.get("config_path", None)
    acoustic_model_path = kwargs["acoustic_model_path"]
    corpus_directory = kwargs["corpus_directory"].absolute()
    dictionary_path = kwargs["dictionary_path"]
    output_directory = kwargs["output_directory"]
    trainer = DictionaryTrainer(
        acoustic_model_path=acoustic_model_path,
        corpus_directory=corpus_directory,
        dictionary_path=dictionary_path,
        **DictionaryTrainer.parse_parameters(config_path, context.params, context.args),
    )

    try:
        trainer.align()
        trainer.export_lexicons(output_directory)
    except Exception:
        trainer.dirty = True
        raise
    finally:
        trainer.cleanup()
