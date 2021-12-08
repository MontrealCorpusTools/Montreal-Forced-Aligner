"""Command line functions for training dictionaries with pronunciation probabilities"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

from montreal_forced_aligner.alignment.pretrained import DictionaryTrainer
from montreal_forced_aligner.command_line.utils import validate_model_arg
from montreal_forced_aligner.exceptions import ArgumentError

if TYPE_CHECKING:
    from argparse import Namespace


__all__ = ["train_dictionary", "validate_args", "run_train_dictionary"]


def train_dictionary(args: Namespace, unknown_args: Optional[List[str]] = None) -> None:
    """
    Run the pronunciation probability training

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Command line arguments
    unknown_args: list[str]
        Optional arguments that will be passed to configuration objects
    """
    aligner = DictionaryTrainer(
        acoustic_model_path=args.acoustic_model_path,
        corpus_directory=args.corpus_directory,
        dictionary_path=args.dictionary_path,
        temporary_directory=args.temporary_directory,
        **DictionaryTrainer.parse_parameters(args.config_path, args, unknown_args),
    )

    try:
        aligner.align()
        aligner.export_lexicons(args.output_directory)
    except Exception:
        aligner.dirty = True
        raise
    finally:
        aligner.cleanup()


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
    args.output_directory = args.output_directory.rstrip("/").rstrip("\\")
    args.corpus_directory = args.corpus_directory.rstrip("/").rstrip("\\")
    if not os.path.exists(args.corpus_directory):
        raise ArgumentError(f"Could not find the corpus directory {args.corpus_directory}.")
    if not os.path.isdir(args.corpus_directory):
        raise ArgumentError(
            f"The specified corpus directory ({args.corpus_directory}) is not a directory."
        )

    args.dictionary_path = validate_model_arg(args.dictionary_path, "dictionary")
    args.acoustic_model_path = validate_model_arg(args.acoustic_model_path, "acoustic")


def run_train_dictionary(args: Namespace, unknown: Optional[List[str]] = None) -> None:
    """
    Wrapper function for running pronunciation probability training

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Parsed command line arguments
    unknown: list[str]
        Parsed command line arguments to be passed to the configuration objects
    """
    validate_args(args)
    train_dictionary(args, unknown)
