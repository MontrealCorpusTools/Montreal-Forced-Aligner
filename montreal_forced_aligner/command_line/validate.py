"""Command line functions for validating corpora"""
from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, List, Optional

from montreal_forced_aligner.command_line.utils import validate_model_arg
from montreal_forced_aligner.exceptions import ArgumentError
from montreal_forced_aligner.validation.corpus_validator import (
    PretrainedValidator,
    TrainingValidator,
)
from montreal_forced_aligner.validation.dictionary_validator import DictionaryValidator

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


def validate_dictionary(args: Namespace, unknown_args: Optional[List[str]] = None) -> None:
    """
    Run the validation command

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Command line arguments
    unknown_args: list[str]
        Optional arguments that will be passed to configuration objects
    """
    validator = DictionaryValidator(
        g2p_model_path=args.g2p_model_path,
        dictionary_path=args.dictionary_path,
        temporary_directory=args.temporary_directory,
        **DictionaryValidator.parse_parameters(args.config_path, args, unknown_args),
    )
    try:
        validator.validate(output_path=args.output_path)
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


def validate_dictionary_args(args: Namespace) -> None:
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
    if sys.platform == "win32":
        raise ArgumentError(
            "Cannot validate dictionaries on native Windows, please use Windows Subsystem for Linux."
        )

    args.dictionary_path = validate_model_arg(args.dictionary_path, "dictionary")
    if args.g2p_model_path:
        args.g2p_model_path = validate_model_arg(args.g2p_model_path, "g2p")


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


def run_validate_dictionary(args: Namespace, unknown: Optional[List[str]] = None) -> None:
    """
    Wrapper function for running dictionary validation

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Parsed command line arguments
    unknown: list[str]
        Parsed command line arguments to be passed to the configuration objects
    """
    validate_dictionary_args(args)
    validate_dictionary(args, unknown)
