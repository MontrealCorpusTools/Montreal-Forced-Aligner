"""Command line functions for training ivector extractors"""
from __future__ import annotations

import os
import shutil
import time
from typing import TYPE_CHECKING, Optional

from montreal_forced_aligner.aligner import PretrainedAligner
from montreal_forced_aligner.command_line.utils import validate_model_arg
from montreal_forced_aligner.config import (
    TEMP_DIR,
    load_basic_train_ivector,
    load_command_configuration,
    train_yaml_to_config,
)
from montreal_forced_aligner.corpus import Corpus
from montreal_forced_aligner.dictionary import MultispeakerDictionary
from montreal_forced_aligner.exceptions import ArgumentError
from montreal_forced_aligner.models import AcousticModel
from montreal_forced_aligner.utils import log_config, setup_logger

if TYPE_CHECKING:
    from argparse import Namespace

__all__ = ["train_ivector", "validate_args", "run_train_ivector_extractor"]


def train_ivector(args: Namespace, unknown_args: Optional[list] = None) -> None:
    """
    Run the ivector extractor training

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Command line arguments
    unknown_args: List[str]
        Optional arguments that will be passed to configuration objects
    """
    from montreal_forced_aligner.utils import get_mfa_version

    command = "train_ivector"
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
    if args.config_path:
        train_config, align_config, dictionary_config = train_yaml_to_config(args.config_path)
    else:
        train_config, align_config, dictionary_config = load_basic_train_ivector()
    if unknown_args:
        train_config.update_from_unknown_args(unknown_args)
        align_config.update_from_unknown_args(unknown_args)
    train_config.use_mp = not args.disable_mp
    align_config.use_mp = not args.disable_mp
    conf_path = os.path.join(data_directory, "config.yml")
    if getattr(args, "clean", False) and os.path.exists(data_directory):
        print("Cleaning old directory!")
        shutil.rmtree(data_directory, ignore_errors=True)
    if getattr(args, "verbose", False):
        log_level = "debug"
    else:
        log_level = "info"
    logger = setup_logger(command, data_directory, console_level=log_level)
    logger.debug("TRAIN CONFIG:")
    log_config(logger, train_config)
    logger.debug("ALIGN CONFIG:")
    log_config(logger, align_config)

    conf = load_command_configuration(
        conf_path,
        {
            "dirty": False,
            "begin": all_begin,
            "version": get_mfa_version(),
            "type": command,
            "corpus_directory": args.corpus_directory,
            "dictionary_path": args.dictionary_path,
            "acoustic_model_path": args.acoustic_model_path,
        },
    )
    if (
        conf["dirty"]
        or conf["type"] != command
        or conf["corpus_directory"] != args.corpus_directory
        or conf["version"] != get_mfa_version()
        or conf["dictionary_path"] != args.dictionary_path
        or conf["acoustic_model_path"] != args.acoustic_model_path
    ):
        logger.warning(
            "WARNING: Using old temp directory, this might not be ideal for you, use the --clean flag to ensure no "
            "weird behavior for previous versions of the temporary directory."
        )
        if conf["dirty"]:
            logger.debug("Previous run ended in an error (maybe ctrl-c?)")
        if conf["type"] != command:
            logger.debug(
                f"Previous run was a different subcommand than {command} (was {conf['type']})"
            )
        if conf["corpus_directory"] != args.corpus_directory:
            logger.debug(
                "Previous run used source directory "
                f"path {conf['corpus_directory']} (new run: {args.corpus_directory})"
            )
        if conf["version"] != get_mfa_version():
            logger.debug(
                f"Previous run was on {conf['version']} version (new run: {get_mfa_version()})"
            )
        if conf["dictionary_path"] != args.dictionary_path:
            logger.debug(
                f"Previous run used dictionary path {conf['dictionary_path']} "
                f"(new run: {args.dictionary_path})"
            )
        if conf["acoustic_model_path"] != args.acoustic_model_path:
            logger.debug(
                f"Previous run used acoustic model path {conf['acoustic_model_path']} "
                f"(new run: {args.acoustic_model_path})"
            )

    os.makedirs(data_directory, exist_ok=True)
    model_directory = os.path.join(data_directory, "acoustic_models")
    try:
        begin = time.time()
        corpus = Corpus(
            args.corpus_directory,
            data_directory,
            dictionary_config,
            speaker_characters=args.speaker_characters,
            num_jobs=args.num_jobs,
            sample_rate=align_config.feature_config.sample_frequency,
            debug=getattr(args, "debug", False),
            logger=logger,
            use_mp=align_config.use_mp,
        )
        dictionary = MultispeakerDictionary(
            args.dictionary_path,
            data_directory,
            dictionary_config,
            word_set=corpus.word_set,
            logger=logger,
        )
        acoustic_model = AcousticModel(args.acoustic_model_path, root_directory=model_directory)
        acoustic_model.log_details(logger)
        acoustic_model.validate(dictionary)
        a = PretrainedAligner(
            corpus,
            dictionary,
            acoustic_model,
            align_config,
            temp_directory=data_directory,
            logger=logger,
        )
        logger.debug(f"Setup pretrained aligner in {time.time() - begin} seconds")
        a.verbose = args.verbose
        begin = time.time()
        a.align()
        logger.debug(f"Performed alignment in {time.time() - begin} seconds")
        for identifier, trainer in train_config.items():
            trainer.logger = logger
            if identifier != "ivector":
                continue
            begin = time.time()
            trainer.init_training(identifier, data_directory, corpus, dictionary, a)
            trainer.train()
            logger.debug(f"Training took {time.time() - begin} seconds")
            trainer.save(args.output_model_path, root_directory=model_directory)

        logger.info("All done!")
        logger.debug(f"Done! Everything took {time.time() - all_begin} seconds")
    except Exception as e:
        conf["dirty"] = True
        raise e
    finally:
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
        conf.save(conf_path)


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

    args.dictionary_path = validate_model_arg(args.dictionary_path, "dictionary")
    args.acoustic_model_path = validate_model_arg(args.acoustic_model_path, "acoustic")


def run_train_ivector_extractor(args: Namespace, unknown: Optional[list] = None) -> None:
    """
    Wrapper function for running ivector extraction training

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Parsed command line arguments
    unknown: List[str]
        Parsed command line arguments to be passed to the configuration objects
    """
    validate_args(args)
    train_ivector(args, unknown)
