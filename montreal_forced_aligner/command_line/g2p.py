"""Command line functions for generating pronunciations using G2P models"""
from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, Optional

from montreal_forced_aligner.command_line.utils import validate_model_arg
from montreal_forced_aligner.config import TEMP_DIR
from montreal_forced_aligner.config.g2p_config import g2p_yaml_to_config, load_basic_g2p_config
from montreal_forced_aligner.corpus import Corpus
from montreal_forced_aligner.g2p.generator import PyniniDictionaryGenerator as Generator
from montreal_forced_aligner.models import G2PModel
from montreal_forced_aligner.utils import setup_logger

if TYPE_CHECKING:
    from argparse import Namespace


__all__ = ["generate_dictionary", "validate_args", "run_g2p"]


def generate_dictionary(args: Namespace, unknown_args: Optional[list] = None) -> None:
    """
    Run the G2P command

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Command line arguments
    unknown_args: List[str]
        Optional arguments that will be passed to configuration objects
    """
    command = "g2p"
    if not args.temp_directory:
        temp_dir = TEMP_DIR
        temp_dir = os.path.join(temp_dir, "G2P")
    else:
        temp_dir = os.path.expanduser(args.temp_directory)
    if args.clean:
        shutil.rmtree(os.path.join(temp_dir, "G2P"), ignore_errors=True)
        shutil.rmtree(os.path.join(temp_dir, "models", "G2P"), ignore_errors=True)
    if args.config_path:
        g2p_config, dictionary_config = g2p_yaml_to_config(args.config_path)
    else:
        g2p_config, dictionary_config = load_basic_g2p_config()
    g2p_config.use_mp = not args.disable_mp
    if unknown_args:
        g2p_config.update_from_unknown_args(unknown_args)
    if os.path.isdir(args.input_path):
        input_dir = os.path.expanduser(args.input_path)
        corpus_name = os.path.basename(args.input_path)
        if corpus_name == "":
            args.input_path = os.path.dirname(args.input_path)
            corpus_name = os.path.basename(args.input_path)
        data_directory = os.path.join(temp_dir, corpus_name)
        if getattr(args, "verbose", False):
            log_level = "debug"
        else:
            log_level = "info"
        logger = setup_logger(command, data_directory, console_level=log_level)

        corpus = Corpus(
            input_dir,
            data_directory,
            dictionary_config=dictionary_config,
            num_jobs=args.num_jobs,
            use_mp=g2p_config.use_mp,
            parse_text_only_files=True,
        )

        word_set = corpus.word_set
        if not args.include_bracketed:
            word_set = [x for x in word_set if not dictionary_config.check_bracketed(x)]
    else:

        if getattr(args, "verbose", False):
            log_level = "debug"
        else:
            log_level = "info"
        logger = setup_logger(command, temp_dir, console_level=log_level)
        word_set = []
        with open(args.input_path, "r", encoding="utf8") as f:
            for line in f:
                word_set.extend(line.strip().split())
        if not args.include_bracketed:
            word_set = [x for x in word_set if not dictionary_config.check_bracketed(x)]

    logger.info(
        f"Generating transcriptions for the {len(word_set)} word types found in the corpus..."
    )
    if args.g2p_model_path is not None:
        model = G2PModel(
            args.g2p_model_path, root_directory=os.path.join(temp_dir, "models", "G2P")
        )
        model.validate(word_set)
        num_jobs = args.num_jobs
        if not g2p_config.use_mp:
            num_jobs = 1
        gen = Generator(
            model,
            word_set,
            temp_directory=temp_dir,
            num_jobs=num_jobs,
            num_pronunciations=g2p_config.num_pronunciations,
            logger=logger,
        )
        gen.output(args.output_path)
        model.clean_up()
    else:
        with open(args.output_path, "w", encoding="utf8") as f:
            for word in word_set:
                pronunciation = list(word)
                f.write(f"{word} {' '.join(pronunciation)}\n")


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


def run_g2p(args: Namespace, unknown: Optional[list] = None) -> None:
    """
    Wrapper function for running G2P

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Parsed command line arguments
    unknown: List[str]
        Parsed command line arguments to be passed to the configuration objects
    """
    validate_args(args)
    generate_dictionary(args, unknown)
