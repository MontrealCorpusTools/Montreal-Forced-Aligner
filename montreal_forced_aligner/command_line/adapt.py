"""Command line functions for adapting acoustic models to new data"""
from __future__ import annotations

import os
import shutil
import time
from typing import TYPE_CHECKING, Collection, Optional

from montreal_forced_aligner.aligner import AdaptingAligner, PretrainedAligner, TrainableAligner
from montreal_forced_aligner.command_line.utils import validate_model_arg
from montreal_forced_aligner.config import (
    TEMP_DIR,
    align_yaml_to_config,
    load_basic_align,
    load_command_configuration,
)
from montreal_forced_aligner.corpus import Corpus
from montreal_forced_aligner.dictionary import MultispeakerDictionary
from montreal_forced_aligner.exceptions import ArgumentError
from montreal_forced_aligner.models import AcousticModel
from montreal_forced_aligner.utils import get_mfa_version, log_config, setup_logger

if TYPE_CHECKING:
    from argparse import Namespace

__all__ = ["adapt_model", "validate_args", "run_adapt_model"]


def adapt_model(args: Namespace, unknown_args: Optional[Collection[str]] = None) -> None:
    """
    Run the acoustic model adaptation

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Command line arguments
    unknown_args: List[str]
        Optional arguments that will be passed to configuration objects
    """
    command = "adapt"
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
        align_config, dictionary_config = align_yaml_to_config(args.config_path)
    else:
        align_config, dictionary_config = load_basic_align()
    align_config.update_from_args(args)
    if unknown_args:
        align_config.update_from_unknown_args(unknown_args)
    conf_path = os.path.join(data_directory, "config.yml")
    if getattr(args, "clean", False) and os.path.exists(data_directory):
        print("Cleaning old directory!")
        shutil.rmtree(data_directory, ignore_errors=True)
    if getattr(args, "verbose", False):
        log_level = "debug"
    else:
        log_level = "info"
    logger = setup_logger(command, data_directory, console_level=log_level)
    logger.debug("ALIGN CONFIG:")
    log_config(logger, align_config)
    logger.debug("DICTIONARY CONFIG:")
    log_config(logger, dictionary_config)
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
    os.makedirs(model_directory, exist_ok=True)
    acoustic_model = AcousticModel(args.acoustic_model_path, root_directory=model_directory)
    dictionary_config.update(acoustic_model.meta)
    acoustic_model.log_details(logger)
    debug = getattr(args, "debug", False)
    audio_dir = None
    if args.audio_directory:
        audio_dir = args.audio_directory
    try:
        corpus = Corpus(
            args.corpus_directory,
            data_directory,
            dictionary_config,
            speaker_characters=args.speaker_characters,
            num_jobs=args.num_jobs,
            sample_rate=align_config.feature_config.sample_frequency,
            logger=logger,
            use_mp=align_config.use_mp,
            audio_directory=audio_dir,
        )
        logger.info(corpus.speaker_utterance_info())
        dictionary = MultispeakerDictionary(
            args.dictionary_path,
            data_directory,
            dictionary_config,
            logger=logger,
        )
        acoustic_model.validate(dictionary)

        begin = time.time()
        previous = PretrainedAligner(
            corpus,
            dictionary,
            acoustic_model,
            align_config,
            temp_directory=data_directory,
            debug=debug,
            logger=logger,
        )
        if args.full_train:
            training_config, dictionary = acoustic_model.adaptation_config()
            training_config.training_configs[0].update(
                {"beam": align_config.beam, "retry_beam": align_config.retry_beam}
            )
            training_config.update_from_align(align_config)
            logger.debug("ADAPT TRAINING CONFIG:")
            log_config(logger, training_config)
            a = TrainableAligner(
                corpus,
                dictionary,
                training_config,
                align_config,
                temp_directory=data_directory,
                debug=debug,
                logger=logger,
                pretrained_aligner=previous,
            )
            logger.debug(f"Setup adapter trainer in {time.time() - begin} seconds")
            a.verbose = args.verbose
            generate_final_alignments = True
            if args.output_directory is None:
                generate_final_alignments = False
            else:
                os.makedirs(args.output_directory, exist_ok=True)

            begin = time.time()
            a.train(generate_final_alignments)
            logger.debug(f"Trained adapted model in {time.time() - begin} seconds")
            if args.output_model_path is not None:
                a.save(args.output_model_path, root_directory=model_directory)

            if generate_final_alignments:
                a.export_textgrids(args.output_directory)

            a.save(args.output_model_path, root_directory=model_directory)
        else:
            a = AdaptingAligner(
                corpus,
                dictionary,
                previous,
                align_config,
                temp_directory=data_directory,
                debug=debug,
                logger=logger,
            )
            logger.debug(f"Setup adapter trainer in {time.time() - begin} seconds")
            a.verbose = args.verbose
            generate_final_alignments = True
            if args.output_directory is None:
                generate_final_alignments = False
            else:
                os.makedirs(args.output_directory, exist_ok=True)
            begin = time.time()
            a.train()
            logger.debug(f"Mapped adapted model in {time.time() - begin} seconds")
            if args.output_model_path is not None:
                a.save(args.output_model_path, root_directory=model_directory)
            if generate_final_alignments:
                begin = time.time()
                a.align()
                logger.debug(
                    f"Generated alignments with adapted model in {time.time() - begin} seconds"
                )

            if generate_final_alignments:
                a.export_textgrids(args.output_directory)

            a.save(args.output_model_path, root_directory=model_directory)

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
    if not os.path.exists(args.corpus_directory):
        raise ArgumentError(f"Could not find the corpus directory {args.corpus_directory}.")
    if not os.path.isdir(args.corpus_directory):
        raise ArgumentError(
            f"The specified corpus directory ({args.corpus_directory}) is not a directory."
        )

    args.dictionary_path = validate_model_arg(args.dictionary_path, "dictionary")
    args.acoustic_model_path = validate_model_arg(args.acoustic_model_path, "acoustic")


def run_adapt_model(args: Namespace, unknown_args: Optional[Collection] = None) -> None:
    """
    Wrapper function for running acoustic model adaptation

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Parsed command line arguments
    unknown: List[str]
        Parsed command line arguments to be passed to the configuration objects
    """
    validate_args(args)
    adapt_model(args, unknown_args)
