"""Command line functions for training G2P models"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from montreal_forced_aligner.command_line.utils import validate_model_arg
from montreal_forced_aligner.g2p.trainer import PyniniTrainer

if TYPE_CHECKING:
    from argparse import Namespace


__all__ = ["train_g2p", "validate_args", "run_train_g2p"]


def train_g2p(args: Namespace, unknown_args: Optional[List[str]] = None) -> None:
    """
    Run the G2P model training

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Command line arguments
    unknown_args: list[str]
        Optional arguments that will be passed to configuration objects
    """

    trainer = PyniniTrainer(
        dictionary_path=args.dictionary_path,
        temporary_directory=args.temporary_directory,
        **PyniniTrainer.parse_parameters(args.config_path, args, unknown_args)
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
    args.dictionary_path = validate_model_arg(args.dictionary_path, "dictionary")


def run_train_g2p(args: Namespace, unknown: Optional[List[str]] = None) -> None:
    """
    Wrapper function for running G2P model training

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Parsed command line arguments
    unknown: list[str]
        Parsed command line arguments to be passed to the configuration objects
    """
    validate_args(args)
    train_g2p(args, unknown)
