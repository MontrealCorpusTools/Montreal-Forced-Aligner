"""Command line functions for training new acoustic models"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

from montreal_forced_aligner.acoustic_modeling import TrainableAligner
from montreal_forced_aligner.command_line.utils import validate_model_arg
from montreal_forced_aligner.exceptions import ArgumentError

if TYPE_CHECKING:
    from argparse import Namespace


__all__ = ["train_acoustic_model", "validate_args", "run_train_acoustic_model"]


def train_acoustic_model(args: Namespace, unknown_args: Optional[List[str]] = None) -> None:
    """
    Run the acoustic model training

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Command line arguments
    unknown_args: list[str]
        Optional arguments that will be passed to configuration objects
    """
    trainer = TrainableAligner(
        corpus_directory=args.corpus_directory,
        dictionary_path=args.dictionary_path,
        temporary_directory=args.temporary_directory,
        **TrainableAligner.parse_parameters(args.config_path, args, unknown_args),
    )

    try:
        generate_final_alignments = True
        if args.output_directory is None:
            generate_final_alignments = False
        else:
            os.makedirs(args.output_directory, exist_ok=True)

        trainer.train(generate_final_alignments)
        if args.output_model_path is not None:
            trainer.export_model(args.output_model_path)

        if args.output_directory is not None:
            trainer.export_files(args.output_directory)
    except Exception:
        trainer.dirty = True
        raise
    finally:
        trainer.cleanup()


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
    if args.corpus_directory == args.output_directory:
        raise ArgumentError("Corpus directory and output directory cannot be the same folder.")
    if not os.path.exists(args.corpus_directory):
        raise (ArgumentError(f'Could not find the corpus directory "{args.corpus_directory}".'))
    if not os.path.isdir(args.corpus_directory):
        raise (
            ArgumentError(
                f'The specified corpus directory "{args.corpus_directory}" is not a directory.'
            )
        )

    args.dictionary_path = validate_model_arg(args.dictionary_path, "dictionary")


def run_train_acoustic_model(args: Namespace, unknown_args: Optional[List[str]] = None) -> None:
    """
    Wrapper function for running acoustic model training

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Parsed command line arguments
    unknown: list[str]
        Parsed command line arguments to be passed to the configuration objects
    """
    validate_args(args)
    train_acoustic_model(args, unknown_args)
