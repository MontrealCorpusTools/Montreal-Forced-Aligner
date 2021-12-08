"""Command line functions for segmenting audio files"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

from montreal_forced_aligner.exceptions import ArgumentError
from montreal_forced_aligner.segmenter import Segmenter

if TYPE_CHECKING:
    from argparse import Namespace


__all__ = ["create_segments", "validate_args", "run_create_segments"]


def create_segments(args: Namespace, unknown_args: Optional[List[str]] = None) -> None:
    """
    Run the sound file segmentation

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Command line arguments
    unknown_args: list[str]
        Optional arguments that will be passed to configuration objects
    """

    segmenter = Segmenter(
        corpus_directory=args.corpus_directory,
        temporary_directory=args.temporary_directory,
        **Segmenter.parse_parameters(args.config_path, args, unknown_args),
    )
    try:
        segmenter.segment()
        segmenter.export_files(args.output_directory)
    except Exception:
        segmenter.dirty = True
        raise
    finally:
        segmenter.cleanup()


def validate_args(args: Namespace) -> None:
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
    if not os.path.exists(args.corpus_directory):
        raise ArgumentError(f"Could not find the corpus directory {args.corpus_directory}.")
    if not os.path.isdir(args.corpus_directory):
        raise ArgumentError(
            f"The specified corpus directory ({args.corpus_directory}) is not a directory."
        )

    if args.corpus_directory == args.output_directory:
        raise ArgumentError("Corpus directory and output directory cannot be the same folder.")


def run_create_segments(args: Namespace, unknown: Optional[List[str]] = None) -> None:
    """
    Wrapper function for running sound file segmentation

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Parsed command line arguments
    unknown: list[str]
        Parsed command line arguments to be passed to the configuration objects
    """
    validate_args(args)
    create_segments(args, unknown)
