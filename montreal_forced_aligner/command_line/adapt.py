"""Command line functions for adapting acoustic models to new data"""
from __future__ import annotations

import os
import typing
from pathlib import Path

import rich_click as click

from montreal_forced_aligner import config
from montreal_forced_aligner.alignment import AdaptingAligner
from montreal_forced_aligner.command_line.utils import (
    common_options,
    validate_acoustic_model,
    validate_corpus_directory,
    validate_dictionary,
)

__all__ = ["adapt_model_cli"]


@click.command(
    name="adapt",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Adapt an acoustic model",
)
@click.argument("corpus_directory", type=click.UNPROCESSED, callback=validate_corpus_directory)
@click.argument("dictionary_path", type=click.UNPROCESSED, callback=validate_dictionary)
@click.argument("acoustic_model_path", type=click.UNPROCESSED, callback=validate_acoustic_model)
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
@click.option(
    "--no_tokenization",
    is_flag=True,
    help="Flag to disable any pretrained tokenization.",
    default=False,
)
@click.option(
    "--reference_directory",
    help="Directory containing gold standard alignments to use in training",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--custom_mapping_path",
    help="YAML file for mapping phones from acoustic model phone set to phone set in golden alignments.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def adapt_model_cli(context, **kwargs) -> None:
    """
    Adapt an acoustic model to a new corpus.
    """
    if kwargs.get("profile", None) is not None:
        os.environ[config.MFA_PROFILE_VARIABLE] = kwargs.pop("profile")
    config.update_configuration(kwargs)
    config_path: typing.Optional[Path] = kwargs.get("config_path", None)
    output_directory: typing.Optional[Path] = kwargs.get("output_directory", None)
    output_model_path: typing.Optional[Path] = kwargs.get("output_model_path", None)
    corpus_directory = kwargs["corpus_directory"].absolute()
    dictionary_path = kwargs["dictionary_path"]
    acoustic_model_path = kwargs["acoustic_model_path"]
    output_format = kwargs["output_format"]
    include_original_text = kwargs["include_original_text"]
    extra_kwargs = AdaptingAligner.parse_parameters(config_path, context.params, context.args)
    no_tokenization = kwargs["no_tokenization"]
    if no_tokenization:
        extra_kwargs["language"] = "unknown"
    reference_directory: typing.Optional[Path] = kwargs.get("reference_directory", None)
    custom_mapping_path: typing.Optional[Path] = kwargs.get("custom_mapping_path", None)
    adapter = AdaptingAligner(
        corpus_directory=corpus_directory,
        dictionary_path=dictionary_path,
        acoustic_model_path=acoustic_model_path,
        reference_directory=reference_directory,
        custom_mapping_path=custom_mapping_path,
        **extra_kwargs,
    )

    try:
        adapter.adapt()
        if output_directory is not None:
            os.makedirs(output_directory, exist_ok=True)
            adapter.align()
            adapter.analyze_alignments()
            adapter.export_files(
                output_directory,
                output_format,
                include_original_text=include_original_text,
            )
        if output_model_path is not None:
            adapter.export_model(output_model_path)
    except Exception:
        adapter.dirty = True
        raise
    finally:
        adapter.cleanup()
