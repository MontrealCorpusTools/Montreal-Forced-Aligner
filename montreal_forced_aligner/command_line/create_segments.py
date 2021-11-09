"""Command line functions for segmenting audio files"""
from __future__ import annotations

import os
import shutil
import time
from typing import TYPE_CHECKING, Optional

from montreal_forced_aligner.config import (
    TEMP_DIR,
    load_basic_segmentation,
    load_command_configuration,
    segmentation_yaml_to_config,
)
from montreal_forced_aligner.corpus import Corpus
from montreal_forced_aligner.exceptions import ArgumentError
from montreal_forced_aligner.segmenter import Segmenter
from montreal_forced_aligner.utils import log_config, setup_logger

if TYPE_CHECKING:
    from argparse import Namespace


__all__ = ["create_segments", "validate_args", "run_create_segments"]


def create_segments(args: Namespace, unknown_args: Optional[list] = None) -> None:
    """
    Run the sound file segmentation

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Command line arguments
    unknown_args: List[str]
        Optional arguments that will be passed to configuration objects
    """
    from montreal_forced_aligner.utils import get_mfa_version

    command = "create_segments"
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
    conf_path = os.path.join(data_directory, "config.yml")
    if args.config_path:
        segmentation_config = segmentation_yaml_to_config(args.config_path)
    else:
        segmentation_config = load_basic_segmentation()
    segmentation_config.update_from_args(args)
    if unknown_args:
        segmentation_config.update_from_unknown_args(unknown_args)
    if getattr(args, "clean", False) and os.path.exists(data_directory):
        print("Cleaning old directory!")
        shutil.rmtree(data_directory, ignore_errors=True)
    if getattr(args, "verbose", False):
        log_level = "debug"
    else:
        log_level = "info"
    logger = setup_logger(command, data_directory, console_level=log_level)
    log_config(logger, segmentation_config)
    conf = load_command_configuration(
        conf_path,
        {
            "dirty": False,
            "begin": time.time(),
            "version": get_mfa_version(),
            "type": command,
            "corpus_directory": args.corpus_directory,
        },
    )
    if (
        conf["dirty"]
        or conf["type"] != command
        or conf["corpus_directory"] != args.corpus_directory
        or conf["version"] != get_mfa_version()
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

    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(args.output_directory, exist_ok=True)
    try:
        corpus = Corpus(
            args.corpus_directory,
            data_directory,
            sample_rate=segmentation_config.feature_config.sample_frequency,
            num_jobs=args.num_jobs,
            logger=logger,
            use_mp=segmentation_config.use_mp,
            ignore_speakers=True,
        )

        begin = time.time()
        a = Segmenter(
            corpus,
            segmentation_config,
            temp_directory=data_directory,
            debug=getattr(args, "debug", False),
            logger=logger,
        )
        logger.debug(f"Setup segmenter in {time.time() - begin} seconds")
        a.verbose = args.verbose

        begin = time.time()
        a.segment()
        logger.debug(f"Performed segmentation in {time.time() - begin} seconds")

        begin = time.time()
        a.export_segments(args.output_directory)
        logger.debug(f"Exported segmentation in {time.time() - begin} seconds")
        logger.info("Done!")
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
    args.output_directory = args.output_directory.rstrip("/").rstrip("\\")
    args.corpus_directory = args.corpus_directory.rstrip("/").rstrip("\\")
    if not os.path.exists(args.corpus_directory):
        raise ArgumentError(f"Could not find the corpus directory {args.corpus_directory}.")
    if not os.path.isdir(args.corpus_directory):
        raise ArgumentError(
            f"The specified corpus directory ({args.corpus_directory}) is not a directory."
        )

    if args.corpus_directory == args.output_directory:
        raise ArgumentError("Corpus directory and output directory cannot be the same folder.")


def run_create_segments(args: Namespace, unknown: Optional[list] = None) -> None:
    """
    Wrapper function for running sound file segmentation

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Parsed command line arguments
    unknown: List[str]
        Parsed command line arguments to be passed to the configuration objects
    """
    validate_args(args)
    create_segments(args, unknown)
