"""Command line functions for validating corpora"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

from montreal_forced_aligner.command_line.utils import validate_model_arg
from montreal_forced_aligner.exceptions import ArgumentError
from montreal_forced_aligner.validator import PretrainedValidator, TrainingValidator

if TYPE_CHECKING:
    from argparse import Namespace


__all__ = ["validate_corpus", "validate_args", "run_validate_corpus"]


def validate_corpus(args: Namespace, unknown_args: Optional[List[str]] = None) -> None:
    """
    Run the validation command

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Command line arguments
    unknown_args: list[str]
        Optional arguments that will be passed to configuration objects
    """
    if args.acoustic_model_path:
        validator = PretrainedValidator(
            acoustic_model_path=args.acoustic_model_path,
            corpus_directory=args.corpus_directory,
            dictionary_path=args.dictionary_path,
            temporary_directory=args.temporary_directory,
            **PretrainedValidator.parse_parameters(args.config_path, args, unknown_args),
        )
    else:
        validator = TrainingValidator(
            corpus_directory=args.corpus_directory,
            dictionary_path=args.dictionary_path,
            temporary_directory=args.temporary_directory,
            **TrainingValidator.parse_parameters(args.config_path, args, unknown_args),
        )
    try:
        validator.validate()
    except Exception:
        validator.dirty = True
        raise
    finally:
        validator.cleanup()


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
    if args.test_transcriptions and args.ignore_acoustics:
        raise ArgumentError("Cannot test transcriptions without acoustic feature generation.")
    if not os.path.exists(args.corpus_directory):
        raise (ArgumentError(f"Could not find the corpus directory {args.corpus_directory}."))
    if not os.path.isdir(args.corpus_directory):
        raise (
            ArgumentError(
                f"The specified corpus directory ({args.corpus_directory}) is not a directory."
            )
        )

    args.dictionary_path = validate_model_arg(args.dictionary_path, "dictionary")
    if args.acoustic_model_path:
        args.acoustic_model_path = validate_model_arg(args.acoustic_model_path, "acoustic")


def run_validate_corpus(args: Namespace, unknown: Optional[List[str]] = None) -> None:
    """
    Wrapper function for running corpus validation

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Parsed command line arguments
    unknown: list[str]
        Parsed command line arguments to be passed to the configuration objects
    """
    validate_args(args)
    validate_corpus(args, unknown)
