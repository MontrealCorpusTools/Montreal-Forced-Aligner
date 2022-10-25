"""Command line functions for classifying speakers"""
from __future__ import annotations

import os

import click

from montreal_forced_aligner.command_line.utils import (
    check_databases,
    common_options,
    validate_ivector_extractor,
)
from montreal_forced_aligner.config import GLOBAL_CONFIG, MFA_PROFILE_VARIABLE
from montreal_forced_aligner.speaker_classifier import SpeakerClassifier

__all__ = ["classify_speakers_cli"]


@click.command(
    name="diarize",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Diarize a corpus",
)
@click.argument("corpus_directory", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument(
    "ivector_extractor_path", type=click.UNPROCESSED, callback=validate_ivector_extractor
)
@click.argument("output_directory", type=click.Path(file_okay=False, dir_okay=True))
@click.option(
    "--config_path",
    "-c",
    help="Path to config file to use for training.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--num_speakers", "-s", help="Number of speakers if known.", type=int, default=0)
@click.option(
    "--output_format",
    help="Format for aligned output files (default is long_textgrid).",
    default="long_textgrid",
    type=click.Choice(["long_textgrid", "short_textgrid", "json", "csv"]),
)
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def classify_speakers_cli(context, **kwargs) -> None:  # pragma: no cover
    """
    Use an ivector extractor to cluster utterances into speakers
    """
    os.putenv(MFA_PROFILE_VARIABLE, kwargs.get("profile", "global"))
    GLOBAL_CONFIG.current_profile.update(kwargs)
    GLOBAL_CONFIG.save()
    check_databases()
    config_path = kwargs.get("config_path", None)
    corpus_directory = kwargs["corpus_directory"]
    ivector_extractor_path = kwargs["ivector_extractor_path"]
    output_directory = kwargs["output_directory"]
    classifier = SpeakerClassifier(
        corpus_directory=corpus_directory,
        ivector_extractor_path=ivector_extractor_path,
        **SpeakerClassifier.parse_parameters(config_path, context.params, context.args),
    )
    try:

        classifier.cluster_utterances()

        classifier.export_files(output_directory)
    except Exception:
        classifier.dirty = True
        raise
    finally:
        classifier.cleanup()
