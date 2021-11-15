"""Command line functions for validating corpora"""
from __future__ import annotations

import os
import shutil
import time
from typing import TYPE_CHECKING, List, Optional

from montreal_forced_aligner.command_line.utils import validate_model_arg
from montreal_forced_aligner.config import TEMP_DIR
from montreal_forced_aligner.config.dictionary_config import DictionaryConfig
from montreal_forced_aligner.corpus import Corpus
from montreal_forced_aligner.dictionary import MultispeakerDictionary
from montreal_forced_aligner.exceptions import ArgumentError
from montreal_forced_aligner.models import AcousticModel
from montreal_forced_aligner.utils import setup_logger
from montreal_forced_aligner.validator import CorpusValidator

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
    unknown_args: List[str]
        Optional arguments that will be passed to configuration objects
    """
    command = "validate"
    all_begin = time.time()
    if not args.temp_directory:
        temp_dir = TEMP_DIR
    else:
        temp_dir = os.path.expanduser(args.temp_directory)
    corpus_name = os.path.basename(args.corpus_directory)
    if corpus_name == "":
        args.corpus_directory = os.path.dirname(args.corpus_directory)
        corpus_name = os.path.basename(args.corpus_directory)
    data_directory = os.path.join(temp_dir, corpus_name)
    shutil.rmtree(data_directory, ignore_errors=True)

    os.makedirs(data_directory, exist_ok=True)
    model_directory = os.path.join(data_directory, "acoustic_models")
    os.makedirs(model_directory, exist_ok=True)
    if getattr(args, "verbose", False):
        log_level = "debug"
    else:
        log_level = "info"
    logger = setup_logger(command, data_directory, console_level=log_level)
    dictionary_config = DictionaryConfig()
    acoustic_model = None
    if args.acoustic_model_path:
        acoustic_model = AcousticModel(args.acoustic_model_path, root_directory=model_directory)
        acoustic_model.log_details(logger)
        dictionary_config.update(acoustic_model.meta)
    dictionary = MultispeakerDictionary(
        args.dictionary_path,
        data_directory,
        dictionary_config,
        logger=logger,
    )
    if acoustic_model:
        acoustic_model.validate(dictionary)

    corpus = Corpus(
        args.corpus_directory,
        data_directory,
        dictionary_config,
        speaker_characters=args.speaker_characters,
        num_jobs=getattr(args, "num_jobs", 3),
        logger=logger,
        use_mp=not args.disable_mp,
    )
    a = CorpusValidator(
        corpus,
        dictionary,
        temp_directory=data_directory,
        ignore_acoustics=getattr(args, "ignore_acoustics", False),
        test_transcriptions=getattr(args, "test_transcriptions", False),
        use_mp=not args.disable_mp,
        logger=logger,
    )
    begin = time.time()
    a.validate()
    logger.debug(f"Validation took {time.time() - begin} seconds")
    logger.info("All done!")
    logger.debug(f"Done! Everything took {time.time() - all_begin} seconds")
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


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
    unknown: List[str]
        Parsed command line arguments to be passed to the configuration objects
    """
    validate_args(args)
    validate_corpus(args, unknown)
