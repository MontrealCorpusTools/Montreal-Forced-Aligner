"""Command line functions for segmenting audio files"""
from __future__ import annotations

from pathlib import Path

import rich_click as click

from montreal_forced_aligner import config
from montreal_forced_aligner.command_line.utils import (
    common_options,
    validate_acoustic_model,
    validate_dictionary,
)
from montreal_forced_aligner.vad.segmenter import TranscriptionSegmenter, VadSegmenter

__all__ = ["create_segments_vad_cli", "create_segments_cli"]


@click.command(
    name="segment_vad",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Split long audio files into shorter segments",
)
@click.argument(
    "corpus_directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
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
    "--output_format",
    help="Format for aligned output files (default is long_textgrid).",
    default="long_textgrid",
    type=click.Choice(["long_textgrid", "short_textgrid", "json", "csv"]),
)
@click.option(
    "--speechbrain/--no_speechbrain",
    "speechbrain",
    help="Flag for using SpeechBrain's pretrained VAD model",
)
@click.option(
    "--cuda/--no_cuda",
    "cuda",
    help="Flag for using CUDA for SpeechBrain's model",
)
@click.option(
    "--segment_transcripts/--no_segment_transcripts",
    "segment_transcripts",
    help="Flag for using CUDA for SpeechBrain's model",
)
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def create_segments_vad_cli(context, **kwargs) -> None:
    """
    Create segments based on SpeechBrain's voice activity detection (VAD) model or a basic energy-based algorithm
    """
    if kwargs.get("profile", None) is not None:
        config.profile = kwargs.pop("profile")
    config.update_configuration(kwargs)

    config_path = kwargs.get("config_path", None)
    corpus_directory = kwargs["corpus_directory"].absolute()
    output_directory = kwargs["output_directory"]
    output_format = kwargs["output_format"]

    segmenter = VadSegmenter(
        corpus_directory=corpus_directory,
        **VadSegmenter.parse_parameters(config_path, context.params, context.args),
    )
    try:
        segmenter.segment()
        segmenter.export_files(output_directory, output_format)
    except Exception:
        segmenter.dirty = True
        raise
    finally:
        segmenter.cleanup()


@click.command(
    name="segment",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Split long audio files into shorter segments",
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
    "--output_format",
    help="Format for aligned output files (default is long_textgrid).",
    default="long_textgrid",
    type=click.Choice(["long_textgrid", "short_textgrid", "json", "csv"]),
)
@click.option(
    "--cuda/--no_cuda",
    "cuda",
    help="Flag for using CUDA for SpeechBrain's model",
)
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def create_segments_cli(context, **kwargs) -> None:
    """
    Create segments based on SpeechBrain's voice activity detection (VAD) model or a basic energy-based algorithm
    """
    if kwargs.get("profile", None) is not None:
        config.profile = kwargs.pop("profile")
    config.update_configuration(kwargs)

    config_path = kwargs.get("config_path", None)
    corpus_directory = kwargs["corpus_directory"].absolute()
    dictionary_path = kwargs["dictionary_path"]
    acoustic_model_path = kwargs["acoustic_model_path"]
    output_directory = kwargs["output_directory"]
    output_format = kwargs["output_format"]

    segmenter = TranscriptionSegmenter(
        corpus_directory=corpus_directory,
        dictionary_path=dictionary_path,
        acoustic_model_path=acoustic_model_path,
        **TranscriptionSegmenter.parse_parameters(config_path, context.params, context.args),
    )
    try:
        segmenter.segment()
        segmenter.export_files(output_directory, output_format)
    except Exception:
        segmenter.dirty = True
        raise
    finally:
        segmenter.cleanup()
