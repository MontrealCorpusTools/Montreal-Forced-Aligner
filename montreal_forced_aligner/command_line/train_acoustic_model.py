"""Command line functions for training new acoustic models"""
from __future__ import annotations

import os
from pathlib import Path

import click

from montreal_forced_aligner.acoustic_modeling import TrainableAligner
from montreal_forced_aligner.command_line.utils import (
    check_databases,
    cleanup_databases,
    common_options,
    validate_dictionary,
)
from montreal_forced_aligner.config import GLOBAL_CONFIG, MFA_PROFILE_VARIABLE

__all__ = ["train_acoustic_model_cli"]


@click.command(
    name="train",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Train a new acoustic model",
)
@click.argument(
    "corpus_directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument("dictionary_path", type=click.UNPROCESSED, callback=validate_dictionary)
@click.argument(
    "output_model_path", type=click.Path(file_okay=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--output_directory",
    help="Path to save alignments.",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
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
@click.option(
    "--phone_set",
    "phone_set_type",
    help="Enable extra decision tree modeling based on the phone set.",
    default="UNKNOWN",
    type=click.Choice(["UNKNOWN", "AUTO", "MFA", "IPA", "ARPA", "PINYIN"]),
)
@click.option(
    "--phone_groups_path",
    "phone_groups_path",
    help="Path to yaml file defining phone groups.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--rules_path",
    "rules_path",
    help="Path to yaml file defining phonological rules.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output_format",
    help="Format for aligned output files (default is long_textgrid).",
    default="long_textgrid",
    type=click.Choice(["long_textgrid", "short_textgrid", "json", "csv"]),
)
@click.option(
    "--include_original_text",
    is_flag=True,
    help="Flag to include original utterance text in the output.",
    default=False,
)
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def train_acoustic_model_cli(context, **kwargs) -> None:
    """
    Train a new acoustic model on a corpus and optionally export alignments
    """
    if kwargs.get("profile", None) is not None:
        os.putenv(MFA_PROFILE_VARIABLE, kwargs["profile"])
    GLOBAL_CONFIG.current_profile.update(kwargs)
    GLOBAL_CONFIG.save()
    check_databases()
    config_path = kwargs.get("config_path", None)
    output_model_path = kwargs.get("output_model_path", None)
    output_directory = kwargs.get("output_directory", None)
    corpus_directory = kwargs["corpus_directory"]
    dictionary_path = kwargs["dictionary_path"]
    trainer = TrainableAligner(
        corpus_directory=corpus_directory,
        dictionary_path=dictionary_path,
        **TrainableAligner.parse_parameters(config_path, context.params, context.args),
    )
    try:
        trainer.train()
        if output_model_path is not None:
            trainer.export_model(output_model_path)

        if output_directory is not None:
            trainer.export_files(
                output_directory,
                kwargs["output_format"],
                include_original_text=kwargs["include_original_text"],
            )
    except Exception:
        trainer.dirty = True
        raise
    finally:
        trainer.cleanup()
        cleanup_databases()
