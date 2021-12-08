"""Command line functions for training language models"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

from montreal_forced_aligner.command_line.utils import validate_model_arg
from montreal_forced_aligner.exceptions import ArgumentError
from montreal_forced_aligner.language_modeling.trainer import (
    LmArpaTrainer,
    LmCorpusTrainer,
    LmDictionaryCorpusTrainer,
)

if TYPE_CHECKING:
    from argparse import Namespace

__all__ = ["train_lm", "validate_args", "run_train_lm"]


def train_lm(args: Namespace, unknown_args: Optional[List[str]] = None) -> None:
    """
    Run the language model training

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Command line arguments
    unknown_args: list[str]
        Optional arguments that will be passed to configuration objects
    """

    if not args.source_path.lower().endswith(".arpa"):
        if not args.dictionary_path:
            trainer = LmCorpusTrainer(
                corpus_directory=args.source_path,
                temporary_directory=args.temporary_directory,
                **LmCorpusTrainer.parse_parameters(args.config_path, args, unknown_args),
            )
        else:
            trainer = LmDictionaryCorpusTrainer(
                corpus_directory=args.source_path,
                dictionary_path=args.dictionary_path,
                temporary_directory=args.temporary_directory,
                **LmDictionaryCorpusTrainer.parse_parameters(args.config_path, args, unknown_args),
            )
    else:
        trainer = LmArpaTrainer(
            arpa_path=args.source_path,
            temporary_directory=args.temporary_directory,
            **LmArpaTrainer.parse_parameters(args.config_path, args, unknown_args),
        )

    try:
        trainer.setup()
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
    args.source_path = args.source_path.rstrip("/").rstrip("\\")
    if args.dictionary_path:
        args.dictionary_path = validate_model_arg(args.dictionary_path, "dictionary")
    if not args.source_path.endswith(".arpa"):
        if not os.path.exists(args.source_path):
            raise (ArgumentError(f"Could not find the corpus directory {args.source_path}."))
        if not os.path.isdir(args.source_path):
            raise (
                ArgumentError(
                    f"The specified corpus directory ({args.source_path}) is not a directory."
                )
            )
    else:
        if not os.path.exists(args.source_path):
            raise (ArgumentError(f"Could not find the source file {args.source_path}."))
    if args.config_path and not os.path.exists(args.config_path):
        raise (ArgumentError(f"Could not find the config file {args.config_path}."))
    if args.model_path and not os.path.exists(args.model_path):
        raise (ArgumentError(f"Could not find the model file {args.model_path}."))


def run_train_lm(args: Namespace, unknown: Optional[List[str]] = None) -> None:
    """
    Wrapper function for running language model training

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Parsed command line arguments
    unknown: list[str]
        Parsed command line arguments to be passed to the configuration objects
    """
    validate_args(args)
    train_lm(args, unknown)
