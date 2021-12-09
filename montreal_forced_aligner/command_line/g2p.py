"""Command line functions for generating pronunciations using G2P models"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

from montreal_forced_aligner.command_line.utils import validate_model_arg
from montreal_forced_aligner.g2p.generator import (
    OrthographicCorpusGenerator,
    OrthographicWordListGenerator,
    PyniniCorpusGenerator,
    PyniniWordListGenerator,
)

if TYPE_CHECKING:
    from argparse import Namespace


__all__ = ["generate_dictionary", "validate_args", "run_g2p"]


def generate_dictionary(args: Namespace, unknown_args: Optional[List[str]] = None) -> None:
    """
    Run the G2P command

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Command line arguments
    unknown_args: list[str]
        Optional arguments that will be passed to configuration objects
    """

    if args.g2p_model_path is None:
        if os.path.isdir(args.input_path):
            g2p = OrthographicCorpusGenerator(
                corpus_directory=args.input_path,
                temporary_directory=args.temporary_directory,
                **OrthographicCorpusGenerator.parse_parameters(
                    args.config_path, args, unknown_args
                )
            )
        else:
            g2p = OrthographicWordListGenerator(
                word_list_path=args.input_path,
                temporary_directory=args.temporary_directory,
                **OrthographicWordListGenerator.parse_parameters(
                    args.config_path, args, unknown_args
                )
            )

    else:
        if os.path.isdir(args.input_path):
            g2p = PyniniCorpusGenerator(
                g2p_model_path=args.g2p_model_path,
                corpus_directory=args.input_path,
                temporary_directory=args.temporary_directory,
                **PyniniCorpusGenerator.parse_parameters(args.config_path, args, unknown_args)
            )
        else:
            g2p = PyniniWordListGenerator(
                g2p_model_path=args.g2p_model_path,
                word_list_path=args.input_path,
                temporary_directory=args.temporary_directory,
                **PyniniWordListGenerator.parse_parameters(args.config_path, args, unknown_args)
            )

    try:
        g2p.setup()
        g2p.export_pronunciations(args.output_path)
    except Exception:
        g2p.dirty = True
        raise
    finally:
        g2p.cleanup()


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
    if not args.g2p_model_path:
        args.g2p_model_path = None
    else:
        args.g2p_model_path = validate_model_arg(args.g2p_model_path, "g2p")


def run_g2p(args: Namespace, unknown: Optional[List[str]] = None) -> None:
    """
    Wrapper function for running G2P

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Parsed command line arguments
    unknown: list[str]
        Parsed command line arguments to be passed to the configuration objects
    """
    validate_args(args)
    generate_dictionary(args, unknown)
