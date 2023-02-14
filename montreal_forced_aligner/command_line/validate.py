"""Command line functions for validating corpora"""
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
)
from montreal_forced_aligner.config import GLOBAL_CONFIG, MFA_PROFILE_VARIABLE
from montreal_forced_aligner.validation.corpus_validator import (
    PretrainedValidator,
    TrainingValidator,
)
from montreal_forced_aligner.validation.dictionary_validator import DictionaryValidator

__all__ = ["validate_corpus_cli", "validate_dictionary_cli"]


@click.command(
    name="validate",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Validate corpus",
)
@click.argument(
    "corpus_directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument("dictionary_path", type=click.UNPROCESSED, callback=validate_dictionary)
@click.option(
    "--acoustic_model_path",
    help="Acoustic model to use in testing alignments.",
    type=click.UNPROCESSED,
    callback=validate_acoustic_model,
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
    "--phone_set",
    help="Enable extra decision tree modeling based on the phone set.",
    default="UNKNOWN",
    type=click.Choice(["UNKNOWN", "AUTO", "IPA", "ARPA", "PINYIN"]),
)
@click.option(
    "--ignore_acoustics",
    "--skip_acoustics",
    "ignore_acoustics",
    is_flag=True,
    help="Skip acoustic feature generation and associated validation.",
    default=False,
)
@click.option(
    "--test_transcriptions",
    is_flag=True,
    help="Use per-speaker language models to test accuracy of transcriptions.",
    default=False,
)
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def validate_corpus_cli(context, **kwargs) -> None:
    """
    Validate a corpus for use in MFA.
    """
    if kwargs.get("profile", None) is not None:
        os.putenv(MFA_PROFILE_VARIABLE, kwargs["profile"])
    GLOBAL_CONFIG.current_profile.update(kwargs)
    GLOBAL_CONFIG.save()
    check_databases()
    config_path = kwargs.get("config_path", None)
    corpus_directory = kwargs["corpus_directory"]
    dictionary_path = kwargs["dictionary_path"]
    acoustic_model_path = kwargs.get("acoustic_model_path", None)
    if acoustic_model_path:
        validator = PretrainedValidator(
            corpus_directory=corpus_directory,
            dictionary_path=dictionary_path,
            acoustic_model_path=acoustic_model_path,
            **PretrainedValidator.parse_parameters(config_path, context.params, context.args),
        )
    else:
        validator = TrainingValidator(
            corpus_directory=corpus_directory,
            dictionary_path=dictionary_path,
            **TrainingValidator.parse_parameters(config_path, context.params, context.args),
        )
    try:
        validator.validate()
    except Exception:
        validator.dirty = True
        raise
    finally:
        validator.cleanup()
        cleanup_databases()


@click.command(
    name="validate_dictionary",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Validate dictionary",
)
@click.argument("dictionary_path", type=str)
@click.option(
    "--output_path",
    help="Path to save the CSV file with the scored pronunciations.",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
@click.option("--g2p_model_path", help="Pretrained G2P model path.", type=str)
@click.option(
    "--config_path",
    "-c",
    help="Path to config file to use for training.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--g2p_threshold",
    help="Threshold to use when running G2P. "
    "Paths with costs less than the best path times the threshold value will be included.",
    type=float,
    default=1.5,
)
@common_options
@click.help_option("-h", "--help")
def validate_dictionary_cli(*args, **kwargs) -> None:
    """
    Validate a dictionary using a G2P model to detect unlikely pronunciations.
    """
    if kwargs.get("profile", None) is not None:
        os.putenv(MFA_PROFILE_VARIABLE, kwargs["profile"])
        GLOBAL_CONFIG.current_profile.update(kwargs)
        GLOBAL_CONFIG.save()
    check_databases()
    config_path = kwargs.get("config_path", None)
    g2p_model_path = kwargs["g2p_model_path"]
    dictionary_path = kwargs["dictionary_path"]
    output_path = kwargs["output_path"]
    validator = DictionaryValidator(
        g2p_model_path=g2p_model_path,
        dictionary_path=dictionary_path,
        **DictionaryValidator.parse_parameters(config_path, kwargs, args),
    )
    try:
        validator.validate(output_path=output_path)
    except Exception:
        validator.dirty = True
        raise
    finally:
        validator.cleanup()
        cleanup_databases()
