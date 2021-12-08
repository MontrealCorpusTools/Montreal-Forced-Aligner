"""Command line functions for adapting acoustic models to new data"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, List, Optional

from montreal_forced_aligner.alignment import AdaptingAligner
from montreal_forced_aligner.command_line.utils import validate_model_arg
from montreal_forced_aligner.exceptions import ArgumentError

if TYPE_CHECKING:
    from argparse import Namespace

__all__ = ["adapt_model", "validate_args", "run_adapt_model"]


def adapt_model(args: Namespace, unknown_args: Optional[List[str]] = None) -> None:
    """
    Run the acoustic model adaptation

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Command line arguments
    unknown_args: list[str]
        Optional arguments that will be passed to configuration objects
    """
    adapter = AdaptingAligner(
        acoustic_model_path=args.acoustic_model_path,
        corpus_directory=args.corpus_directory,
        dictionary_path=args.dictionary_path,
        temporary_directory=args.temporary_directory,
        **AdaptingAligner.parse_parameters(args.config_path, args, unknown_args),
    )

    try:
        adapter.adapt()
        generate_final_alignments = True
        if args.output_directory is None:
            generate_final_alignments = False
        else:
            os.makedirs(args.output_directory, exist_ok=True)
        export_model = True
        if args.output_model_path is None:
            export_model = False

        if generate_final_alignments:
            begin = time.time()
            adapter.align()
            adapter.logger.debug(
                f"Generated alignments with adapted model in {time.time() - begin} seconds"
            )
            adapter.export_files(args.output_directory)
        if export_model:
            adapter.export_model(args.output_model_path)
    except Exception:
        adapter.dirty = True
        raise
    finally:
        adapter.cleanup()


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
    try:
        args.speaker_characters = int(args.speaker_characters)
    except ValueError:
        pass

    args.output_directory = None
    if not args.output_model_path:
        args.output_model_path = None
    output_paths = args.output_paths
    if len(output_paths) > 2:
        raise ArgumentError(f"Got more arguments for output_paths than 2: {output_paths}")
    for path in output_paths:
        if path.endswith(".zip"):
            args.output_model_path = path
        else:
            args.output_directory = path.rstrip("/").rstrip("\\")

    args.corpus_directory = args.corpus_directory.rstrip("/").rstrip("\\")
    if not os.path.exists(args.corpus_directory):
        raise ArgumentError(f"Could not find the corpus directory {args.corpus_directory}.")
    if not os.path.isdir(args.corpus_directory):
        raise ArgumentError(
            f"The specified corpus directory ({args.corpus_directory}) is not a directory."
        )

    args.dictionary_path = validate_model_arg(args.dictionary_path, "dictionary")
    args.acoustic_model_path = validate_model_arg(args.acoustic_model_path, "acoustic")


def run_adapt_model(args: Namespace, unknown_args: Optional[List[str]] = None) -> None:
    """
    Wrapper function for running acoustic model adaptation

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Parsed command line arguments
    unknown: list[str]
        Parsed command line arguments to be passed to the configuration objects
    """
    validate_args(args)
    adapt_model(args, unknown_args)
