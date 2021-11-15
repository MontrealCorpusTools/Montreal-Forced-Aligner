"""Command line functions for training new acoustic models"""
from __future__ import annotations

import os
import shutil
import time
from typing import TYPE_CHECKING, Optional

from montreal_forced_aligner.aligner import TrainableAligner
from montreal_forced_aligner.command_line.utils import validate_model_arg
from montreal_forced_aligner.config import (
    TEMP_DIR,
    load_basic_train,
    load_command_configuration,
    train_yaml_to_config,
)
from montreal_forced_aligner.corpus import Corpus
from montreal_forced_aligner.dictionary import MultispeakerDictionary
from montreal_forced_aligner.exceptions import ArgumentError
from montreal_forced_aligner.utils import log_config, setup_logger

if TYPE_CHECKING:
    from argparse import Namespace


__all__ = ["train_acoustic_model", "validate_args", "run_train_acoustic_model"]


def train_acoustic_model(args: Namespace, unknown_args: Optional[list] = None) -> None:
    """
    Run the acoustic model training

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Command line arguments
    unknown_args: List[str]
        Optional arguments that will be passed to configuration objects
    """
    from montreal_forced_aligner.utils import get_mfa_version

    command = "train_acoustic_model"
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
        train_config, align_config, dictionary_config = load_basic_train()
    train_config.use_mp = not args.disable_mp
    align_config.use_mp = not args.disable_mp
    align_config.debug = args.debug
    align_config.overwrite = args.overwrite
    align_config.cleanup_textgrids = not args.disable_textgrid_cleanup
    if unknown_args:
        train_config.update_from_unknown_args(unknown_args)
        align_config.update_from_unknown_args(unknown_args)
    train_config.update_from_align(align_config)
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
    if args.debug:
        logger.warning("Running in DEBUG mode, may have impact on performance and disk usage.")
    conf = load_command_configuration(
        conf_path,
        {
            "dirty": False,
            "begin": time.time(),
            "version": get_mfa_version(),
            "type": command,
            "corpus_directory": args.corpus_directory,
            "dictionary_path": args.dictionary_path,
        },
    )
    if (
        conf["dirty"]
        or conf["type"] != command
        or conf["corpus_directory"] != args.corpus_directory
        or conf["version"] != get_mfa_version()
        or conf["dictionary_path"] != args.dictionary_path
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

    os.makedirs(data_directory, exist_ok=True)
    model_directory = os.path.join(data_directory, "acoustic_models")
    audio_dir = None
    if args.audio_directory:
        audio_dir = args.audio_directory
    try:
        corpus = Corpus(
            args.corpus_directory,
            data_directory,
            dictionary_config,
            speaker_characters=args.speaker_characters,
            num_jobs=getattr(args, "num_jobs", 3),
            sample_rate=align_config.feature_config.sample_frequency,
            debug=getattr(args, "debug", False),
            logger=logger,
            use_mp=align_config.use_mp,
            audio_directory=audio_dir,
        )
        logger.info(corpus.speaker_utterance_info())
        dictionary = MultispeakerDictionary(
            args.dictionary_path,
            data_directory,
            dictionary_config,
            word_set=corpus.word_set,
            logger=logger,
        )
        utt_oov_path = os.path.join(corpus.split_directory, "utterance_oovs.txt")
        if os.path.exists(utt_oov_path):
            shutil.copy(utt_oov_path, args.output_directory)
        oov_path = os.path.join(corpus.split_directory, "oovs_found.txt")
        if os.path.exists(oov_path):
            shutil.copy(oov_path, args.output_directory)
        a = TrainableAligner(
            corpus,
            dictionary,
            train_config,
            align_config,
            temp_directory=data_directory,
            logger=logger,
            debug=getattr(args, "debug", False),
        )
        a.verbose = args.verbose
        begin = time.time()
        generate_final_alignments = True
        if args.output_directory is None:
            generate_final_alignments = False
        else:
            os.makedirs(args.output_directory, exist_ok=True)

        a.train(generate_final_alignments)
        logger.debug(f"Training took {time.time() - begin} seconds")
        if args.output_model_path is not None:
            a.save(args.output_model_path, root_directory=model_directory)

        if args.output_directory is not None:
            a.export_textgrids(args.output_directory)
        logger.info("All done!")
        logger.debug(f"Done! Everything took {time.time() - all_begin} seconds")
    except Exception:
        conf["dirty"] = True
        raise
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

    args.output_directory = None
    if not args.output_model_path:
        args.output_model_path = None
    output_paths = args.output_paths
    if len(output_paths) > 2:
        raise ArgumentError(f"Got more arguments for output_paths than 2: {output_paths}")
    for path in output_paths:
        if path.endswith(".zip"):
            args.output_model_path = path
        else:
            args.output_directory = path.rstrip("/").rstrip("\\")

    args.corpus_directory = args.corpus_directory.rstrip("/").rstrip("\\")
    if args.corpus_directory == args.output_directory:
        raise ArgumentError("Corpus directory and output directory cannot be the same folder.")
    if not os.path.exists(args.corpus_directory):
        raise (ArgumentError(f'Could not find the corpus directory "{args.corpus_directory}".'))
    if not os.path.isdir(args.corpus_directory):
        raise (
            ArgumentError(
                f'The specified corpus directory "{args.corpus_directory}" is not a directory.'
            )
        )

    args.dictionary_path = validate_model_arg(args.dictionary_path, "dictionary")


def run_train_acoustic_model(args: Namespace, unknown_args: Optional[list] = None) -> None:
    """
    Wrapper function for running acoustic model training

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Parsed command line arguments
    unknown: List[str]
        Parsed command line arguments to be passed to the configuration objects
    """
    validate_args(args)
    train_acoustic_model(args, unknown_args)
