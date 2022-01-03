"""Command line functions for classifying speakers"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

from montreal_forced_aligner.command_line.utils import validate_model_arg
from montreal_forced_aligner.exceptions import ArgumentError
from montreal_forced_aligner.speaker_classifier import SpeakerClassifier

if TYPE_CHECKING:
    from argparse import Namespace

__all__ = ["classify_speakers", "validate_args", "run_classify_speakers"]


def classify_speakers(
    args: Namespace, unknown_args: Optional[List[str]] = None
) -> None:  # pragma: no cover
    """
    Run the speaker classification

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Command line arguments
    unknown_args: list[str]
        Optional arguments that will be passed to configuration objects
    """
    classifier = SpeakerClassifier(
        ivector_extractor_path=args.ivector_extractor_path,
        corpus_directory=args.corpus_directory,
        temporary_directory=args.temporary_directory,
        **SpeakerClassifier.parse_parameters(args.config_path, args, unknown_args),
    )
    try:

        classifier.cluster_utterances()

        classifier.export_files(args.output_directory)
    except Exception:
        classifier.dirty = True
        raise
    finally:
        classifier.cleanup()


def validate_args(args: Namespace) -> None:  # pragma: no cover
    """
    Validate the command line arguments

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Parsed command line arguments

    Raises
    ------
    :class:`~montreal_forced_aligner.exceptions.ArgumentError`
        If there is a problem with any arguments
    """
    args.output_directory = args.output_directory.rstrip("/").rstrip("\\")
    args.corpus_directory = args.corpus_directory.rstrip("/").rstrip("\\")
    if args.cluster and not args.num_speakers:
        raise ArgumentError("If using clustering, num_speakers must be specified")
    if not os.path.exists(args.corpus_directory):
        raise ArgumentError(f"Could not find the corpus directory {args.corpus_directory}.")
    if not os.path.isdir(args.corpus_directory):
        raise ArgumentError(
            f"The specified corpus directory ({args.corpus_directory}) is not a directory."
        )

    if args.corpus_directory == args.output_directory:
        raise ArgumentError("Corpus directory and output directory cannot be the same folder.")

    args.ivector_extractor_path = validate_model_arg(args.ivector_extractor_path, "ivector")


def run_classify_speakers(
    args: Namespace, unknown: Optional[List[str]] = None
) -> None:  # pragma: no cover
    """
    Wrapper function for running speaker classification

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Parsed command line arguments
    unknown: list[str]
        Parsed command line arguments to be passed to the configuration objects
    """
    validate_args(args)
    classify_speakers(args)
