"""Command line functions for classifying speakers"""
from __future__ import annotations

import os
from pathlib import Path

import click

from montreal_forced_aligner.command_line.utils import (
    check_databases,
    cleanup_databases,
    common_options,
    validate_ivector_extractor,
)
from montreal_forced_aligner.config import GLOBAL_CONFIG, MFA_PROFILE_VARIABLE
from montreal_forced_aligner.data import ClusterType
from montreal_forced_aligner.diarization.speaker_diarizer import SpeakerDiarizer

__all__ = ["diarize_speakers_cli"]


@click.command(
    name="diarize",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Diarize a corpus",
)
@click.argument(
    "corpus_directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument(
    "ivector_extractor_path", type=click.UNPROCESSED, callback=validate_ivector_extractor
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
    "--expected_num_speakers", "-s", help="Number of speakers if known.", type=int, default=0
)
@click.option(
    "--output_format",
    help="Format for aligned output files (default is long_textgrid).",
    default="long_textgrid",
    type=click.Choice(["long_textgrid", "short_textgrid", "json", "csv"]),
)
@click.option(
    "--classify/--cluster",
    "classify",
    is_flag=True,
    default=False,
    help="Specify whether to classify speakers into pretrained IDs or cluster speakers without a classification model, default is cluster",
)
@click.option(
    "--cluster_type",
    help="Type of clustering algorithm to use",
    default=ClusterType.mfa.name,
    type=click.Choice([x.name for x in ClusterType]),
)
@click.option(
    "--cuda/--no_cuda",
    "cuda",
    is_flag=True,
    default=False,
    help="Flag for using CUDA for SpeechBrain's model",
)
@click.option(
    "--use_pca/--no_use_pca",
    "use_pca",
    is_flag=True,
    default=True,
    help="Flag for using PCA representations of ivectors",
)
@click.option(
    "--evaluate",
    "--validate",
    "evaluation_mode",
    is_flag=True,
    help="Flag for whether to evaluate clustering/classification against existing speakers.",
    default=False,
)
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def diarize_speakers_cli(context, **kwargs) -> None:
    """
    Use an ivector extractor to cluster utterances into speakers

    If you would like to use SpeechBrain's speaker recognition model, specify ``speechbrain`` as the ``ivector_extractor_path``.
    When using SpeechBrain's speaker recognition model, the ``--cuda`` flag is available to perform computations on GPU, and
    the ``--num_jobs`` parameter will be used as a the batch size for any parallel computation.
    """
    if kwargs.get("profile", None) is not None:
        os.putenv(MFA_PROFILE_VARIABLE, kwargs["profile"])
    GLOBAL_CONFIG.current_profile.update(kwargs)
    GLOBAL_CONFIG.save()
    check_databases()
    config_path = kwargs.get("config_path", None)
    corpus_directory = kwargs["corpus_directory"]
    ivector_extractor_path = kwargs["ivector_extractor_path"]
    output_directory = kwargs["output_directory"]
    classify = kwargs.get("classify", False)
    classifier = SpeakerDiarizer(
        corpus_directory=corpus_directory,
        ivector_extractor_path=ivector_extractor_path,
        **SpeakerDiarizer.parse_parameters(config_path, context.params, context.args),
    )
    try:
        if classify:
            classifier.classify_speakers()
        else:
            classifier.cluster_utterances()

        classifier.export_files(output_directory)
    except Exception:
        classifier.dirty = True
        raise
    finally:
        classifier.cleanup()
        cleanup_databases()
