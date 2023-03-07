"""Command line functions for generating pronunciations using G2P models"""
from __future__ import annotations

import os
from pathlib import Path

import rich_click as click

from montreal_forced_aligner.command_line.utils import common_options, validate_tokenizer_model
from montreal_forced_aligner.config import GLOBAL_CONFIG, MFA_PROFILE_VARIABLE
from montreal_forced_aligner.tokenization.tokenizer import CorpusTokenizer

__all__ = ["tokenize_cli"]


@click.command(
    name="tokenize",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Tokenize utterances",
)
@click.argument(
    "input_path", type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path)
)
@click.argument("tokenizer_model_path", type=click.UNPROCESSED, callback=validate_tokenizer_model)
@click.argument(
    "output_directory", type=click.Path(file_okay=False, dir_okay=True, path_type=Path)
)
@click.option(
    "--config_path",
    "-c",
    help="Path to config file to use for training.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def tokenize_cli(context, **kwargs) -> None:
    """
    Tokenize utterances with a trained tokenizer model
    """
    if kwargs.get("profile", None) is not None:
        os.environ[MFA_PROFILE_VARIABLE] = kwargs.pop("profile")
    GLOBAL_CONFIG.current_profile.update(kwargs)
    GLOBAL_CONFIG.save()

    config_path = kwargs.get("config_path", None)
    input_path = kwargs["input_path"]
    tokenizer_model_path = kwargs["tokenizer_model_path"]
    output_directory = kwargs["output_directory"]

    tokenizer = CorpusTokenizer(
        corpus_directory=input_path,
        tokenizer_model_path=tokenizer_model_path,
        **CorpusTokenizer.parse_parameters(config_path, context.params, context.args),
    )

    try:
        tokenizer.setup()
        tokenizer.tokenize_utterances()
        tokenizer.export_files(output_directory)
    except Exception:
        tokenizer.dirty = True
        raise
    finally:
        tokenizer.cleanup()
