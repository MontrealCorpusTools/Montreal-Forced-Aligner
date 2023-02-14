"""Command line functions for transcribing corpora"""
from __future__ import annotations

import os
from pathlib import Path

import click

from montreal_forced_aligner.command_line.utils import (
    check_databases,
    cleanup_databases,
    common_options,
    validate_acoustic_model,
    validate_dictionary,
    validate_language_model,
)
from montreal_forced_aligner.config import GLOBAL_CONFIG, MFA_PROFILE_VARIABLE
from montreal_forced_aligner.transcription import Transcriber

__all__ = ["transcribe_corpus_cli"]


@click.command(
    name="transcribe",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Transcribe audio files",
)
@click.argument(
    "corpus_directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument("dictionary_path", type=click.UNPROCESSED, callback=validate_dictionary)
@click.argument("acoustic_model_path", type=click.UNPROCESSED, callback=validate_acoustic_model)
@click.argument("language_model_path", type=click.UNPROCESSED, callback=validate_language_model)
@click.argument(
    "output_directory", type=click.Path(file_okay=False, dir_okay=True, path_type=Path)
)
@click.option(
    "--config_path",
    "-c",
    help="Path to config file to use for training.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--speaker_characters",
    "-s",
    help="Number of characters of file names to use for determining speaker, "
    "default is to use directory names.",
    type=str,
    default="0",
)
@click.option(
    "--audio_directory",
    "-a",
    help="Audio directory root to use for finding audio files.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--output_type",
    help="Flag for outputting transcription text or alignments.",
    default="transcription",
    type=click.Choice(["transcription", "alignment"]),
)
@click.option(
    "--output_format",
    help="Format for aligned output files (default is long_textgrid).",
    default="long_textgrid",
    type=click.Choice(["long_textgrid", "short_textgrid", "json", "csv"]),
)
@click.option(
    "--evaluate",
    "evaluation_mode",
    is_flag=True,
    help="Evaluate the transcription against golden texts.",
    default=False,
)
@click.option(
    "--include_original_text",
    is_flag=True,
    help="Flag to include original utterance text in the output.",
    default=False,
)
@click.option(
    "--language_model_weight",
    help="Specific language model weight to use in evaluating transcriptions, defaults to 16.",
    type=int,
    default=16,
)
@click.option(
    "--word_insertion_penalty",
    help="Specific word insertion penalty between 0.0 and 1.0 to use in evaluating transcription, defaults to 1.0.",
    type=float,
    default=1.0,
)
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def transcribe_corpus_cli(context, **kwargs) -> None:
    """
    Transcribe utterances using an acoustic model, language model, and pronunciation dictionary.
    """
    if kwargs.get("profile", None) is not None:
        os.putenv(MFA_PROFILE_VARIABLE, kwargs["profile"])
    GLOBAL_CONFIG.current_profile.update(kwargs)
    GLOBAL_CONFIG.save()
    check_databases()
    config_path = kwargs.get("config_path", None)
    corpus_directory = kwargs["corpus_directory"]
    acoustic_model_path = kwargs["acoustic_model_path"]
    language_model_path = kwargs["language_model_path"]
    dictionary_path = kwargs["dictionary_path"]
    output_directory = kwargs["output_directory"]
    output_format = kwargs["output_format"]
    include_original_text = kwargs["include_original_text"]
    transcriber = Transcriber(
        corpus_directory=corpus_directory,
        dictionary_path=dictionary_path,
        acoustic_model_path=acoustic_model_path,
        language_model_path=language_model_path,
        **Transcriber.parse_parameters(config_path, context.params, context.args),
    )
    try:
        transcriber.setup()
        transcriber.transcribe()
        transcriber.export_files(
            output_directory,
            output_format=output_format,
            include_original_text=include_original_text,
        )
    except Exception:
        transcriber.dirty = True
        raise
    finally:
        transcriber.cleanup()
        cleanup_databases()
