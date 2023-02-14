"""Command line functions for segmenting audio files"""
from __future__ import annotations

import os
from pathlib import Path

import click

from montreal_forced_aligner.command_line.utils import (
    check_databases,
    cleanup_databases,
    common_options,
)
from montreal_forced_aligner.config import GLOBAL_CONFIG, MFA_PROFILE_VARIABLE
from montreal_forced_aligner.vad.segmenter import Segmenter

__all__ = ["create_segments_cli"]


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
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def create_segments_cli(context, **kwargs) -> None:
    """
    Create segments based on SpeechBrain's voice activity detection (VAD) model or a basic energy-based algorithm
    """
    if kwargs.get("profile", None) is not None:
        os.putenv(MFA_PROFILE_VARIABLE, kwargs["profile"])
    GLOBAL_CONFIG.current_profile.update(kwargs)
    GLOBAL_CONFIG.save()
    check_databases()

    config_path = kwargs.get("config_path", None)
    corpus_directory = kwargs["corpus_directory"]
    output_directory = kwargs["output_directory"]
    output_format = kwargs["output_format"]

    segmenter = Segmenter(
        corpus_directory=corpus_directory,
        **Segmenter.parse_parameters(config_path, context.params, context.args),
    )
    try:
        segmenter.segment()
        segmenter.export_files(output_directory, output_format)
    except Exception:
        segmenter.dirty = True
        raise
    finally:
        segmenter.cleanup()
        cleanup_databases()
