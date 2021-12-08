"""Command line functions for training ivector extractors"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

from montreal_forced_aligner.exceptions import ArgumentError
from montreal_forced_aligner.ivector.trainer import TrainableIvectorExtractor

if TYPE_CHECKING:
    from argparse import Namespace

__all__ = ["train_ivector", "validate_args", "run_train_ivector_extractor"]


def train_ivector(args: Namespace, unknown_args: Optional[List[str]] = None) -> None:
    """
    Run the ivector extractor training

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Command line arguments
    unknown_args: list[str]
        Optional arguments that will be passed to configuration objects
    """

    trainer = TrainableIvectorExtractor(
        corpus_directory=args.corpus_directory,
        temporary_directory=args.temporary_directory,
        **TrainableIvectorExtractor.parse_parameters(args.config_path, args, unknown_args),
    )

    try:

        trainer.train()
        trainer.export_model(args.output_model_path)

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
    args.corpus_directory = args.corpus_directory.rstrip("/").rstrip("\\")
    if args.config_path and not os.path.exists(args.config_path):
        raise (ArgumentError(f"Could not find the config file {args.config_path}."))

    if not os.path.exists(args.corpus_directory):
        raise (ArgumentError(f"Could not find the corpus directory {args.corpus_directory}."))
    if not os.path.isdir(args.corpus_directory):
        raise (
            ArgumentError(
                f"The specified corpus directory ({args.corpus_directory}) is not a directory."
            )
        )


def run_train_ivector_extractor(args: Namespace, unknown: Optional[List[str]] = None) -> None:
    """
    Wrapper function for running ivector extraction training

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Parsed command line arguments
    unknown: list[str]
        Parsed command line arguments to be passed to the configuration objects
    """
    validate_args(args)
    train_ivector(args, unknown)
