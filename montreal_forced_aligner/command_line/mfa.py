"""Command line functions for calling the root mfa command"""
from __future__ import annotations

import argparse
import atexit
import multiprocessing as mp
import sys
import time
from datetime import datetime
from typing import TYPE_CHECKING

from montreal_forced_aligner.command_line.adapt import run_adapt_model
from montreal_forced_aligner.command_line.align import run_align_corpus
from montreal_forced_aligner.command_line.anchor import run_anchor
from montreal_forced_aligner.command_line.classify_speakers import run_classify_speakers
from montreal_forced_aligner.command_line.create_segments import run_create_segments
from montreal_forced_aligner.command_line.g2p import run_g2p
from montreal_forced_aligner.command_line.model import run_model
from montreal_forced_aligner.command_line.train_acoustic_model import run_train_acoustic_model
from montreal_forced_aligner.command_line.train_dictionary import run_train_dictionary
from montreal_forced_aligner.command_line.train_g2p import run_train_g2p
from montreal_forced_aligner.command_line.train_ivector_extractor import (
    run_train_ivector_extractor,
)
from montreal_forced_aligner.command_line.train_lm import run_train_lm
from montreal_forced_aligner.command_line.transcribe import run_transcribe_corpus
from montreal_forced_aligner.command_line.validate import run_validate_corpus
from montreal_forced_aligner.config import (
    load_command_history,
    load_global_config,
    update_command_history,
    update_global_config,
)
from montreal_forced_aligner.exceptions import MFAError
from montreal_forced_aligner.models import MODEL_TYPES
from montreal_forced_aligner.utils import check_third_party

if TYPE_CHECKING:
    from argparse import ArgumentParser


BEGIN = time.time()
BEGIN_DATE = datetime.now()


__all__ = ["ExitHooks", "create_parser", "main"]


class ExitHooks(object):
    """
    Class for capturing exit information for MFA commands
    """

    def __init__(self):
        self.exit_code = None
        self.exception = None

    def hook(self):
        """Hook for capturing information about exit code and exceptions"""
        self._orig_exit = sys.exit
        sys.exit = self.exit
        sys.excepthook = self.exc_handler

    def exit(self, code=0):
        """Actual exit for the program"""
        self.exit_code = code
        self._orig_exit(code)

    def exc_handler(self, exc_type, exc, *args):
        """Handle and save exceptions"""
        self.exception = exc
        self.exit_code = 1

    def history_save_handler(self) -> None:
        """
        Handler for saving history on exit.  In addition to the command run, also saves exit code, whether
        an exception was encountered, when the command was executed, and how long it took to run
        """
        from montreal_forced_aligner.utils import get_mfa_version

        history_data = {
            "command": " ".join(sys.argv),
            "execution_time": time.time() - BEGIN,
            "date": BEGIN_DATE,
            "version": get_mfa_version(),
        }

        if self.exit_code is not None:
            history_data["exit_code"] = self.exit_code
            history_data["exception"] = ""
        elif self.exception is not None:
            history_data["exit_code"] = 1
            history_data["exception"] = str(self.exception)
        else:
            history_data["exception"] = ""
            history_data["exit_code"] = 0
        update_command_history(history_data)
        if self.exception:
            raise self.exception


def create_parser() -> ArgumentParser:
    """
    Constructs the MFA argument parser

    Returns
    -------
    :class:`~argparse.ArgumentParser`
        MFA argument parser
    """
    GLOBAL_CONFIG = load_global_config()

    def add_global_options(subparser: argparse.ArgumentParser, textgrid_output: bool = False):
        """
        Add a set of global options to a subparser

        Parameters
        ----------
        subparser: :class:`~argparse.ArgumentParser`
            Subparser to augment
        textgrid_output: bool
            Flag for whether the subparser is used for a command that generates TextGrids
        """
        subparser.add_argument(
            "-t",
            "--temp_directory",
            "--temporary_directory",
            dest="temporary_directory",
            type=str,
            default=GLOBAL_CONFIG["temporary_directory"],
            help=f"Temporary directory root to store MFA created files, default is {GLOBAL_CONFIG['temporary_directory']}",
        )
        subparser.add_argument(
            "--disable_mp",
            help=f"Disable any multiprocessing during alignment (not recommended), default is {not GLOBAL_CONFIG['use_mp']}",
            action="store_true",
            default=not GLOBAL_CONFIG["use_mp"],
        )
        subparser.add_argument(
            "-j",
            "--num_jobs",
            type=int,
            default=GLOBAL_CONFIG["num_jobs"],
            help=f"Number of data splits (and cores to use if multiprocessing is enabled), defaults "
            f"is {GLOBAL_CONFIG['num_jobs']}",
        )
        subparser.add_argument(
            "-v",
            "--verbose",
            help=f"Output debug messages, default is {GLOBAL_CONFIG['verbose']}",
            action="store_true",
            default=GLOBAL_CONFIG["verbose"],
        )
        subparser.add_argument(
            "--clean",
            help=f"Remove files from previous runs, default is {GLOBAL_CONFIG['clean']}",
            action="store_true",
            default=GLOBAL_CONFIG["clean"],
        )
        subparser.add_argument(
            "--overwrite",
            help=f"Overwrite output files when they exist, default is {GLOBAL_CONFIG['overwrite']}",
            action="store_true",
            default=GLOBAL_CONFIG["overwrite"],
        )
        subparser.add_argument(
            "--debug",
            help=f"Run extra steps for debugging issues, default is {GLOBAL_CONFIG['debug']}",
            action="store_true",
            default=GLOBAL_CONFIG["debug"],
        )
        if textgrid_output:
            subparser.add_argument(
                "--disable_textgrid_cleanup",
                help=f"Disable extra clean up steps on TextGrid output, default is {not GLOBAL_CONFIG['cleanup_textgrids']}",
                action="store_true",
                default=not GLOBAL_CONFIG["cleanup_textgrids"],
            )

    pretrained_acoustic = ", ".join(MODEL_TYPES["acoustic"].get_available_models())
    if not pretrained_acoustic:
        pretrained_acoustic = (
            "you can use ``mfa model download acoustic`` to get pretrained MFA models"
        )

    pretrained_ivector = ", ".join(MODEL_TYPES["ivector"].get_available_models())
    if not pretrained_ivector:
        pretrained_ivector = (
            "you can use ``mfa model download ivector`` to get pretrained MFA models"
        )

    pretrained_g2p = ", ".join(MODEL_TYPES["g2p"].get_available_models())
    if not pretrained_g2p:
        pretrained_g2p = "you can use ``mfa model download g2p`` to get pretrained MFA models"

    pretrained_lm = ", ".join(MODEL_TYPES["language_model"].get_available_models())
    if not pretrained_lm:
        pretrained_lm = (
            "you can use ``mfa model download language_model`` to get pretrained MFA models"
        )

    pretrained_dictionary = ", ".join(MODEL_TYPES["dictionary"].get_available_models())
    if not pretrained_dictionary:
        pretrained_dictionary = (
            "you can use ``mfa model download dictionary`` to get MFA dictionaries"
        )

    dictionary_path_help = f"Full path to pronunciation dictionary, or saved dictionary name ({pretrained_dictionary})"

    acoustic_model_path_help = (
        f"Full path to pre-trained acoustic model, or saved model name ({pretrained_acoustic})"
    )
    language_model_path_help = (
        f"Full path to pre-trained language model, or saved model name ({pretrained_lm})"
    )
    ivector_model_path_help = f"Full path to pre-trained ivector extractor model, or saved model name ({pretrained_ivector})"
    g2p_model_path_help = (
        f"Full path to pre-trained G2P model, or saved model name ({pretrained_g2p}). "
        "If not specified, then orthographic transcription is split into pronunciations."
    )

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="subcommand")
    subparsers.required = True

    _ = subparsers.add_parser("version")

    align_parser = subparsers.add_parser(
        "align", help="Align a corpus with a pretrained acoustic model"
    )
    align_parser.add_argument("corpus_directory", help="Full path to the directory to align")
    align_parser.add_argument(
        "dictionary_path",
        help=dictionary_path_help,
        type=str,
    )
    align_parser.add_argument(
        "acoustic_model_path",
        type=str,
        help=acoustic_model_path_help,
    )
    align_parser.add_argument(
        "output_directory",
        type=str,
        help="Full path to output directory, will be created if it doesn't exist",
    )
    align_parser.add_argument(
        "--config_path", type=str, default="", help="Path to config file to use for alignment"
    )
    align_parser.add_argument(
        "-s",
        "--speaker_characters",
        type=str,
        default="0",
        help="Number of characters of file names to use for determining speaker, "
        "default is to use directory names",
    )
    align_parser.add_argument(
        "-a",
        "--audio_directory",
        type=str,
        default="",
        help="Audio directory root to use for finding audio files",
    )
    align_parser.add_argument(
        "--reference_directory",
        type=str,
        default="",
        help="Directory containing gold standard alignments to evaluate",
    )
    align_parser.add_argument(
        "--custom_mapping_path",
        type=str,
        default="",
        help="YAML file for mapping phones across phone sets in evaluations",
    )
    add_global_options(align_parser, textgrid_output=True)

    adapt_parser = subparsers.add_parser("adapt", help="Adapt an acoustic model to a new corpus")
    adapt_parser.add_argument("corpus_directory", help="Full path to the directory to align")
    adapt_parser.add_argument("dictionary_path", type=str, help=dictionary_path_help)
    adapt_parser.add_argument(
        "acoustic_model_path",
        type=str,
        help=acoustic_model_path_help,
    )
    adapt_parser.add_argument(
        "output_paths",
        type=str,
        nargs="+",
        help="Path to save the new acoustic model, path to export aligned TextGrids, or both",
    )
    adapt_parser.add_argument(
        "-o",
        "--output_model_path",
        type=str,
        default="",
        help="Full path to save adapted acoustic model",
    )
    adapt_parser.add_argument(
        "--config_path", type=str, default="", help="Path to config file to use for alignment"
    )
    adapt_parser.add_argument(
        "-s",
        "--speaker_characters",
        type=str,
        default="0",
        help="Number of characters of file names to use for determining speaker, "
        "default is to use directory names",
    )
    adapt_parser.add_argument(
        "-a",
        "--audio_directory",
        type=str,
        default="",
        help="Audio directory root to use for finding audio files",
    )
    add_global_options(adapt_parser, textgrid_output=True)

    train_parser = subparsers.add_parser(
        "train", help="Train a new acoustic model on a corpus and optionally export alignments"
    )
    train_parser.add_argument(
        "corpus_directory", type=str, help="Full path to the source directory to align"
    )
    train_parser.add_argument("dictionary_path", type=str, help=dictionary_path_help, default="")
    train_parser.add_argument(
        "output_paths",
        type=str,
        nargs="+",
        help="Path to save the new acoustic model, path to export aligned TextGrids, or both",
    )
    train_parser.add_argument(
        "--config_path",
        type=str,
        default="",
        help="Path to config file to use for training and alignment",
    )
    train_parser.add_argument(
        "-o",
        "--output_model_path",
        type=str,
        default="",
        help="Full path to save resulting acoustic model",
    )
    train_parser.add_argument(
        "-s",
        "--speaker_characters",
        type=str,
        default="0",
        help="Number of characters of filenames to use for determining speaker, "
        "default is to use directory names",
    )
    train_parser.add_argument(
        "-a",
        "--audio_directory",
        type=str,
        default="",
        help="Audio directory root to use for finding audio files",
    )
    train_parser.add_argument(
        "--phone_set",
        dest="phone_set_type",
        type=str,
        help="Enable extra decision tree modeling based on the phone set",
        default="UNKNOWN",
        choices=["AUTO", "IPA", "ARPA", "PINYIN"],
    )
    add_global_options(train_parser, textgrid_output=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a corpus for use in MFA")
    validate_parser.add_argument(
        "corpus_directory", type=str, help="Full path to the source directory to align"
    )
    validate_parser.add_argument(
        "dictionary_path", type=str, help=dictionary_path_help, default=""
    )
    validate_parser.add_argument(
        "acoustic_model_path",
        type=str,
        nargs="?",
        default="",
        help=acoustic_model_path_help,
    )
    validate_parser.add_argument(
        "-s",
        "--speaker_characters",
        type=str,
        default="0",
        help="Number of characters of file names to use for determining speaker, "
        "default is to use directory names",
    )
    validate_parser.add_argument(
        "--config_path",
        type=str,
        default="",
        help="Path to config file to use for training and alignment",
    )
    validate_parser.add_argument(
        "--test_transcriptions", help="Test accuracy of transcriptions", action="store_true"
    )
    validate_parser.add_argument(
        "--ignore_acoustics",
        "--skip_acoustics",
        dest="ignore_acoustics",
        help="Skip acoustic feature generation and associated validation",
        action="store_true",
    )
    validate_parser.add_argument(
        "-a",
        "--audio_directory",
        type=str,
        default="",
        help="Audio directory root to use for finding audio files",
    )
    validate_parser.add_argument(
        "--phone_set",
        dest="phone_set_type",
        type=str,
        help="Enable extra decision tree modeling based on the phone set",
        default="UNKNOWN",
        choices=["AUTO", "IPA", "ARPA", "PINYIN"],
    )
    add_global_options(validate_parser)

    g2p_parser = subparsers.add_parser(
        "g2p", help="Generate a pronunciation dictionary using a G2P model"
    )
    g2p_parser.add_argument(
        "g2p_model_path",
        help=g2p_model_path_help,
        type=str,
        nargs="?",
    )

    g2p_parser.add_argument(
        "input_path",
        type=str,
        help="Corpus to base word list on or a text file of words to generate pronunciations",
    )
    g2p_parser.add_argument("output_path", type=str, help="Path to save output dictionary")
    g2p_parser.add_argument(
        "--include_bracketed",
        help="Included words enclosed by brackets, job_name.e. [...], (...), <...>",
        action="store_true",
    )
    g2p_parser.add_argument(
        "--config_path", type=str, default="", help="Path to config file to use for G2P"
    )
    add_global_options(g2p_parser)

    train_g2p_parser = subparsers.add_parser(
        "train_g2p", help="Train a G2P model from a pronunciation dictionary"
    )
    train_g2p_parser.add_argument("dictionary_path", type=str, help=dictionary_path_help)

    train_g2p_parser.add_argument(
        "output_model_path", type=str, help="Desired location of generated model"
    )
    train_g2p_parser.add_argument(
        "--config_path", type=str, default="", help="Path to config file to use for G2P"
    )
    train_g2p_parser.add_argument(
        "--evaluate",
        "--validate",
        dest="evaluation_mode",
        action="store_true",
        help="Perform an analysis of accuracy training on "
        "most of the data and validating on an unseen subset",
    )
    add_global_options(train_g2p_parser)
    help_message = "Inspect, download, and save pretrained MFA models"
    model_parser = subparsers.add_parser(
        "model", aliases=["models"], description=help_message, help=help_message
    )

    model_subparsers = model_parser.add_subparsers(dest="action")
    model_subparsers.required = True
    help_message = "Download a pretrained model from the MFA repository"
    model_download_parser = model_subparsers.add_parser(
        "download", description=help_message, help=help_message
    )
    model_download_parser.add_argument(
        "model_type", choices=sorted(MODEL_TYPES), help="Type of model to download"
    )
    model_download_parser.add_argument(
        "name",
        help="Name of language code to download, if not specified, "
        "will list all available languages",
        type=str,
        nargs="?",
    )
    help_message = "List of saved models"
    model_list_parser = model_subparsers.add_parser(
        "list", description=help_message, help=help_message
    )
    model_list_parser.add_argument(
        "model_type",
        choices=sorted(MODEL_TYPES),
        type=str,
        nargs="?",
        help="Type of model to list",
    )

    help_message = "Inspect a model and output its metadata"
    model_inspect_parser = model_subparsers.add_parser(
        "inspect", description=help_message, help=help_message
    )
    model_inspect_parser.add_argument(
        "model_type",
        choices=sorted(MODEL_TYPES),
        type=str,
        nargs="?",
        help="Type of model to download",
    )
    model_inspect_parser.add_argument(
        "name", type=str, help="Name of pretrained model or path to MFA model to inspect"
    )

    help_message = "Save a MFA model to the pretrained directory for name-based referencing"
    model_save_parser = model_subparsers.add_parser(
        "save", description=help_message, help=help_message
    )
    model_save_parser.add_argument(
        "model_type", type=str, choices=sorted(MODEL_TYPES), help="Type of MFA model"
    )
    model_save_parser.add_argument(
        "path", help="Path to MFA model to save for invoking with just its name"
    )
    model_save_parser.add_argument(
        "--name",
        help="Name to use as reference (defaults to the name of the zip file",
        type=str,
        default="",
    )
    model_save_parser.add_argument(
        "--overwrite",
        help="Flag to overwrite existing pretrained models with the same name (and model type)",
        action="store_true",
    )

    train_lm_parser = subparsers.add_parser(
        "train_lm", help="Train a language model from a corpus"
    )
    train_lm_parser.add_argument(
        "source_path",
        type=str,
        help="Full path to the source directory to train from, alternatively "
        "an ARPA format language model to convert for MFA use",
    )
    train_lm_parser.add_argument(
        "output_model_path", type=str, help="Full path to save resulting language model"
    )
    train_lm_parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        help="Full path to existing language model to merge probabilities",
    )
    train_lm_parser.add_argument(
        "-w",
        "--model_weight",
        type=float,
        default=1.0,
        help="Weight factor for supplemental language model, defaults to 1.0",
    )
    train_lm_parser.add_argument(
        "--dictionary_path", type=str, help=dictionary_path_help, default=""
    )
    train_lm_parser.add_argument(
        "--config_path",
        type=str,
        default="",
        help="Path to config file to use for training and alignment",
    )
    add_global_options(train_lm_parser)

    train_dictionary_parser = subparsers.add_parser(
        "train_dictionary",
        help="Calculate pronunciation probabilities for a dictionary based on alignment results in a corpus",
    )
    train_dictionary_parser.add_argument(
        "corpus_directory", help="Full path to the directory to align"
    )
    train_dictionary_parser.add_argument("dictionary_path", type=str, help=dictionary_path_help)
    train_dictionary_parser.add_argument(
        "acoustic_model_path",
        type=str,
        help=acoustic_model_path_help,
    )
    train_dictionary_parser.add_argument(
        "output_directory",
        type=str,
        help="Full path to output directory, will be created if it doesn't exist",
    )
    train_dictionary_parser.add_argument(
        "--config_path", type=str, default="", help="Path to config file to use for alignment"
    )
    train_dictionary_parser.add_argument(
        "-s",
        "--speaker_characters",
        type=str,
        default="0",
        help="Number of characters of file names to use for determining speaker, "
        "default is to use directory names",
    )
    add_global_options(train_dictionary_parser)

    train_ivector_parser = subparsers.add_parser(
        "train_ivector",
        help="Train an ivector extractor from a corpus and pretrained acoustic model",
    )
    train_ivector_parser.add_argument(
        "corpus_directory",
        type=str,
        help="Full path to the source directory to train the ivector extractor",
    )
    train_ivector_parser.add_argument(
        "output_model_path",
        type=str,
        help="Full path to save resulting ivector extractor",
    )
    train_ivector_parser.add_argument(
        "-s",
        "--speaker_characters",
        type=str,
        default="0",
        help="Number of characters of filenames to use for determining speaker, "
        "default is to use directory names",
    )
    train_ivector_parser.add_argument(
        "--config_path", type=str, default="", help="Path to config file to use for training"
    )
    add_global_options(train_ivector_parser)

    classify_speakers_parser = subparsers.add_parser(
        "classify_speakers", help="Use an ivector extractor to cluster utterances into speakers"
    )
    classify_speakers_parser.add_argument(
        "corpus_directory",
        type=str,
        help="Full path to the source directory to run speaker classification",
    )
    classify_speakers_parser.add_argument(
        "ivector_extractor_path", type=str, default="", help=ivector_model_path_help
    )
    classify_speakers_parser.add_argument(
        "output_directory",
        type=str,
        help="Full path to output directory, will be created if it doesn't exist",
    )

    classify_speakers_parser.add_argument(
        "-s", "--num_speakers", type=int, default=0, help="Number of speakers if known"
    )
    classify_speakers_parser.add_argument(
        "--cluster", help="Using clustering instead of classification", action="store_true"
    )
    classify_speakers_parser.add_argument(
        "--config_path",
        type=str,
        default="",
        help="Path to config file to use for ivector extraction",
    )
    add_global_options(classify_speakers_parser)

    create_segments_parser = subparsers.add_parser(
        "create_segments", help="Create segments based on voice activity dectection (VAD)"
    )
    create_segments_parser.add_argument(
        "corpus_directory", help="Full path to the source directory to run VAD segmentation"
    )
    create_segments_parser.add_argument(
        "output_directory",
        type=str,
        help="Full path to output directory, will be created if it doesn't exist",
    )
    create_segments_parser.add_argument(
        "--config_path", type=str, default="", help="Path to config file to use for segmentation"
    )
    add_global_options(create_segments_parser)

    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe utterances using an acoustic model, language model, and pronunciation dictionary",
    )
    transcribe_parser.add_argument(
        "corpus_directory", type=str, help="Full path to the directory to transcribe"
    )
    transcribe_parser.add_argument("dictionary_path", type=str, help=dictionary_path_help)
    transcribe_parser.add_argument(
        "acoustic_model_path",
        type=str,
        help=acoustic_model_path_help,
    )
    transcribe_parser.add_argument(
        "language_model_path",
        type=str,
        help=language_model_path_help,
    )
    transcribe_parser.add_argument(
        "output_directory",
        type=str,
        help="Full path to output directory, will be created if it doesn't exist",
    )
    transcribe_parser.add_argument(
        "--config_path", type=str, default="", help="Path to config file to use for transcription"
    )
    transcribe_parser.add_argument(
        "-s",
        "--speaker_characters",
        type=str,
        default="0",
        help="Number of characters of file names to use for determining speaker, "
        "default is to use directory names",
    )
    transcribe_parser.add_argument(
        "-a",
        "--audio_directory",
        type=str,
        default="",
        help="Audio directory root to use for finding audio files",
    )
    transcribe_parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluation_mode",
        help="Evaluate the transcription against golden texts",
        action="store_true",
    )
    add_global_options(transcribe_parser)

    config_parser = subparsers.add_parser(
        "configure",
        help="The configure command is used to set global defaults for MFA so "
        "you don't have to set them every time you call an MFA command.",
    )
    config_parser.add_argument(
        "-t",
        "--temp_directory",
        "--temporary_directory",
        dest="temporary_directory",
        type=str,
        default="",
        help=f"Set the default temporary directory, default is {GLOBAL_CONFIG['temporary_directory']}",
    )
    config_parser.add_argument(
        "-j",
        "--num_jobs",
        type=int,
        help=f"Set the number of processes to use by default, defaults to {GLOBAL_CONFIG['num_jobs']}",
    )
    config_parser.add_argument(
        "--always_clean",
        help="Always remove files from previous runs by default",
        action="store_true",
    )
    config_parser.add_argument(
        "--never_clean",
        help="Don't remove files from previous runs by default",
        action="store_true",
    )
    config_parser.add_argument(
        "--always_verbose", help="Default to verbose output", action="store_true"
    )
    config_parser.add_argument(
        "--never_verbose", help="Default to non-verbose output", action="store_true"
    )
    config_parser.add_argument(
        "--always_debug", help="Default to running debugging steps", action="store_true"
    )
    config_parser.add_argument(
        "--never_debug", help="Default to not running debugging steps", action="store_true"
    )
    config_parser.add_argument(
        "--always_overwrite", help="Always overwrite output files", action="store_true"
    )
    config_parser.add_argument(
        "--never_overwrite",
        help="Never overwrite output files (if file already exists, "
        "the output will be saved in the temp directory)",
        action="store_true",
    )
    config_parser.add_argument(
        "--disable_mp",
        help="Disable all multiprocessing (not recommended as it will usually "
        "increase processing times)",
        action="store_true",
    )
    config_parser.add_argument(
        "--enable_mp",
        help="Enable multiprocessing (recommended and enabled by default)",
        action="store_true",
    )
    config_parser.add_argument(
        "--disable_textgrid_cleanup",
        help="Disable postprocessing of TextGrids that cleans up "
        "silences and recombines compound words and clitics",
        action="store_true",
    )
    config_parser.add_argument(
        "--enable_textgrid_cleanup",
        help="Enable postprocessing of TextGrids that cleans up "
        "silences and recombines compound words and clitics",
        action="store_true",
    )
    config_parser.add_argument(
        "--disable_detect_phone_set",
        help="Disable auto-detecting phone sets from the dictionary during training",
        action="store_true",
    )
    config_parser.add_argument(
        "--enable_detect_phone_set",
        help="Enable auto-detecting phone sets from the dictionary during training",
        action="store_true",
    )
    config_parser.add_argument(
        "--disable_terminal_colors", help="Turn off colored text in output", action="store_true"
    )
    config_parser.add_argument(
        "--enable_terminal_colors", help="Turn on colored text in output", action="store_true"
    )
    config_parser.add_argument(
        "--terminal_width",
        help=f"Set width of terminal output, "
        f"currently set to {GLOBAL_CONFIG['terminal_width']}",
        default=GLOBAL_CONFIG["terminal_width"],
        type=int,
    )
    config_parser.add_argument(
        "--blas_num_threads",
        help=f"Number of threads to use for BLAS libraries, 1 is recommended "
        f"due to how much MFA relies on multiprocessing. "
        f"Currently set to {GLOBAL_CONFIG['blas_num_threads']}",
        default=GLOBAL_CONFIG["blas_num_threads"],
        type=int,
    )

    history_parser = subparsers.add_parser("history", help="Show previously run mfa commands")
    _ = subparsers.add_parser("thirdparty", help="DEPRECATED: Please install Kaldi via conda.")
    _ = subparsers.add_parser(
        "download", help="DEPRECATED: Please use mfa model download instead."
    )

    history_parser.add_argument(
        "depth", type=int, help="Number of commands to list", nargs="?", default=10
    )
    history_parser.add_argument(
        "-v",
        "--verbose",
        help=f"Output debug messages, default is {GLOBAL_CONFIG['verbose']}",
        action="store_true",
    )

    _ = subparsers.add_parser(
        "anchor", aliases=["annotator"], help="Launch Anchor Annotator (if installed)"
    )

    return parser


parser = create_parser()


def print_history(args):
    depth = args.depth
    history = load_command_history()[-depth:]
    if args.verbose:
        print("command\tDate\tExecution time\tVersion\tExit code\tException")
        for h in history:
            execution_time = time.strftime("%H:%M:%S", time.gmtime(h["execution_time"]))
            d = h["date"].isoformat()
            print(
                f"{h['command']}\t{d}\t{execution_time}\t{h['version']}\t{h['exit_code']}\t{h['exception']}"
            )
        pass
    else:
        for h in history:
            print(h["command"])


def main() -> None:
    """
    Main function for the MFA command line interface
    """

    check_third_party()

    hooks = ExitHooks()
    hooks.hook()
    atexit.register(hooks.history_save_handler)
    from colorama import init

    init()
    parser = create_parser()
    mp.freeze_support()
    args, unknown = parser.parse_known_args()
    for short in ["-c", "-d"]:
        if short in unknown:
            print(
                f"Due to the number of options that `{short}` could refer to, it is not accepted. "
                "Please specify the full argument",
                file=sys.stderr,
            )
            sys.exit(1)
    try:
        if args.subcommand in ["g2p", "train_g2p"]:
            try:
                import pynini  # noqa
            except ImportError:
                print(
                    "There was an issue importing Pynini, please ensure that it is installed. If you are on Windows, "
                    "please use the Windows Subsystem for Linux to use g2p functionality.",
                    file=sys.stderr,
                )
                sys.exit(1)
        if args.subcommand == "align":
            run_align_corpus(args, unknown)
        elif args.subcommand == "adapt":
            run_adapt_model(args, unknown)
        elif args.subcommand == "train":
            run_train_acoustic_model(args, unknown)
        elif args.subcommand == "g2p":
            run_g2p(args, unknown)
        elif args.subcommand == "train_g2p":
            run_train_g2p(args, unknown)
        elif args.subcommand == "validate":
            run_validate_corpus(args, unknown)
        elif args.subcommand in ["model", "models"]:
            run_model(args)
        elif args.subcommand == "train_lm":
            run_train_lm(args, unknown)
        elif args.subcommand == "train_dictionary":
            run_train_dictionary(args, unknown)
        elif args.subcommand == "train_ivector":
            run_train_ivector_extractor(args, unknown)
        elif args.subcommand == "classify_speakers":  # pragma: no cover
            run_classify_speakers(args, unknown)
        elif args.subcommand in ["annotator", "anchor"]:
            run_anchor()
        elif args.subcommand == "transcribe":
            run_transcribe_corpus(args, unknown)
        elif args.subcommand == "create_segments":
            run_create_segments(args, unknown)
        elif args.subcommand == "configure":
            update_global_config(args)
            global GLOBAL_CONFIG
            GLOBAL_CONFIG = load_global_config()
        elif args.subcommand == "history":
            print_history(args)
        elif args.subcommand == "version":
            from montreal_forced_aligner.utils import get_mfa_version

            print(get_mfa_version())
        elif args.subcommand == "thirdparty":  # Deprecated command
            raise DeprecationWarning(
                "Necessary thirdparty executables are now installed via conda. Please refer to the installation docs for the updated commands."
            )
        elif args.subcommand == "download":  # Deprecated command
            raise DeprecationWarning(
                "Downloading models is now run through the `mfa model download` command, please use that instead."
            )
    except MFAError as e:
        if getattr(args, "debug", False):
            raise
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import warnings

    warnings.warn(
        "Use 'python -m montreal_forced_aligner', not 'python -m montreal_forced_aligner.command_line.mfa'",
        DeprecationWarning,
    )
    main()
