"""Command line functions for training G2P models"""
from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, Optional

from montreal_forced_aligner.command_line.utils import validate_model_arg
from montreal_forced_aligner.config import TEMP_DIR
from montreal_forced_aligner.config.train_g2p_config import (
    load_basic_train_g2p_config,
    train_g2p_yaml_to_config,
)
from montreal_forced_aligner.dictionary import PronunciationDictionary
from montreal_forced_aligner.g2p.trainer import PyniniTrainer as Trainer

if TYPE_CHECKING:
    from argparse import Namespace


__all__ = ["train_g2p", "validate_args", "run_train_g2p"]


def train_g2p(args: Namespace, unknown_args: Optional[list] = None) -> None:
    """
    Run the G2P model training

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Command line arguments
    unknown_args: List[str]
        Optional arguments that will be passed to configuration objects
    """
    if not args.temp_directory:
        temp_dir = TEMP_DIR
    else:
        temp_dir = os.path.expanduser(args.temp_directory)
    if args.clean:
        shutil.rmtree(os.path.join(temp_dir, "G2P"), ignore_errors=True)
        shutil.rmtree(os.path.join(temp_dir, "models", "G2P"), ignore_errors=True)
    if args.config_path:
        train_config, dictionary_config = train_g2p_yaml_to_config(args.config_path)
    else:
        train_config, dictionary_config = load_basic_train_g2p_config()
    train_config.use_mp = not args.disable_mp
    if unknown_args:
        train_config.update_from_unknown_args(unknown_args)
    dictionary = PronunciationDictionary(args.dictionary_path, "", dictionary_config)
    t = Trainer(
        dictionary,
        args.output_model_path,
        temp_directory=temp_dir,
        train_config=train_config,
        num_jobs=args.num_jobs,
        verbose=args.verbose,
    )
    if args.validate:
        t.validate()
    else:
        t.train()


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


def run_train_g2p(args: Namespace, unknown: Optional[list] = None) -> None:
    """
    Wrapper function for running G2P model training

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Parsed command line arguments
    unknown: List[str]
        Parsed command line arguments to be passed to the configuration objects
    """
    validate_args(args)
    train_g2p(args, unknown)
