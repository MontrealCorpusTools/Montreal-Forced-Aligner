"""Command line functions for training G2P models"""
from __future__ import annotations

import os
from pathlib import Path

import rich_click as click

from montreal_forced_aligner.command_line.utils import common_options
from montreal_forced_aligner.config import GLOBAL_CONFIG, MFA_PROFILE_VARIABLE
from montreal_forced_aligner.tokenization.trainer import (
    PhonetisaurusTokenizerTrainer,
    TokenizerTrainer,
)

__all__ = ["train_tokenizer_cli"]


@click.command(
    name="train_tokenizer",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Train a tokenizer model",
)
@click.argument(
    "corpus_directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument(
    "output_model_path", type=click.Path(file_okay=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--config_path",
    "-c",
    help="Path to config file to use for training.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--evaluate",
    "--validate",
    "evaluation_mode",
    is_flag=True,
    help="Perform an analysis of accuracy training on "
    "most of the data and validating on an unseen subset.",
    default=False,
)
@click.option(
    "--phonetisaurus",
    is_flag=True,
    help="Flag for using Phonetisaurus-style models.",
    default=False,
)
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def train_tokenizer_cli(context, **kwargs) -> None:
    """
    Train a tokenizer model from a tokenized corpus.
    """
    if kwargs.get("profile", None) is not None:
        os.environ[MFA_PROFILE_VARIABLE] = kwargs.pop("profile")
    GLOBAL_CONFIG.current_profile.update(kwargs)
    GLOBAL_CONFIG.save()
    config_path = kwargs.get("config_path", None)
    corpus_directory = kwargs["corpus_directory"]
    output_model_path = kwargs["output_model_path"]
    phonetisaurus = kwargs["phonetisaurus"]
    if phonetisaurus:
        trainer = PhonetisaurusTokenizerTrainer(
            corpus_directory=corpus_directory,
            **PhonetisaurusTokenizerTrainer.parse_parameters(
                config_path, context.params, context.args
            ),
        )
    else:
        trainer = TokenizerTrainer(
            corpus_directory=corpus_directory,
            **TokenizerTrainer.parse_parameters(config_path, context.params, context.args),
        )

    try:
        trainer.setup()
        trainer.train()
        trainer.export_model(output_model_path)

    except Exception:
        trainer.dirty = True
        raise
    finally:
        trainer.cleanup()
