"""Command line functions for segmenting audio files"""
from __future__ import annotations

import os

import click

from montreal_forced_aligner.command_line.utils import check_databases, common_options
from montreal_forced_aligner.config import GLOBAL_CONFIG, MFA_PROFILE_VARIABLE
from montreal_forced_aligner.segmenter import Segmenter

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
@click.argument("corpus_directory", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("output_directory", type=click.Path(file_okay=False, dir_okay=True))
@click.option(
    "--config_path",
    "-c",
    help="Path to config file to use for training.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
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
def create_segments_cli(context, **kwargs) -> None:
    """
    Create segments based on voice activity detection (VAD)
    """
    os.putenv(MFA_PROFILE_VARIABLE, kwargs.get("profile", "global"))
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
