"""Command line functions for training G2P models"""
from __future__ import annotations

import os
from pathlib import Path

import click

from montreal_forced_aligner.command_line.utils import (
    check_databases,
    cleanup_databases,
    common_options,
    validate_dictionary,
)
from montreal_forced_aligner.config import GLOBAL_CONFIG, MFA_PROFILE_VARIABLE
from montreal_forced_aligner.g2p.phonetisaurus_trainer import PhonetisaurusTrainer
from montreal_forced_aligner.g2p.trainer import PyniniTrainer

__all__ = ["train_g2p_cli"]


@click.command(
    name="train_g2p",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Train a G2P model",
)
@click.argument("dictionary_path", type=click.UNPROCESSED, callback=validate_dictionary)
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
    "--phonetisaurus",
    is_flag=True,
    help="Flag for using Phonetisaurus-style models.",
    default=False,
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
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def train_g2p_cli(context, **kwargs) -> None:
    """
    Train a G2P model from a pronunciation dictionary.
    """
    if kwargs.get("profile", None) is not None:
        os.putenv(MFA_PROFILE_VARIABLE, kwargs["profile"])
    GLOBAL_CONFIG.current_profile.update(kwargs)
    GLOBAL_CONFIG.save()
    check_databases()
    config_path = kwargs.get("config_path", None)
    dictionary_path = kwargs["dictionary_path"]
    phonetisaurus = kwargs["phonetisaurus"]
    output_model_path = kwargs["output_model_path"]
    if phonetisaurus:
        trainer = PhonetisaurusTrainer(
            dictionary_path=dictionary_path,
            **PhonetisaurusTrainer.parse_parameters(config_path, context.params, context.args),
        )

    else:
        trainer = PyniniTrainer(
            dictionary_path=dictionary_path,
            **PyniniTrainer.parse_parameters(config_path, context.params, context.args),
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
        cleanup_databases()
