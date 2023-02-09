"""Command line functions for training language models"""
from __future__ import annotations

import os

import click

from montreal_forced_aligner.command_line.utils import (
    check_databases,
    cleanup_databases,
    common_options,
    validate_dictionary,
)
from montreal_forced_aligner.config import GLOBAL_CONFIG, MFA_PROFILE_VARIABLE
from montreal_forced_aligner.language_modeling.trainer import (
    MfaLmArpaTrainer,
    MfaLmCorpusTrainer,
    MfaLmDictionaryCorpusTrainer,
)

__all__ = ["train_lm_cli"]


@click.command(
    name="train_lm",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Train a language model",
)
@click.argument("source_path", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.argument("output_model_path", type=click.Path(file_okay=True, dir_okay=False))
@click.option(
    "--dictionary_path",
    help="Full path to pronunciation dictionary, or saved dictionary name.",
    type=click.UNPROCESSED,
    callback=validate_dictionary,
)
@click.option(
    "--config_path",
    "-c",
    help="Path to config file to use for training.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def train_lm_cli(context, **kwargs) -> None:
    """
    Train a language model from a corpus or convert an existing ARPA-format language model to an MFA language model.
    """
    if kwargs.get("profile", None) is not None:
        os.putenv(MFA_PROFILE_VARIABLE, kwargs["profile"])
    GLOBAL_CONFIG.current_profile.update(kwargs)
    GLOBAL_CONFIG.save()
    check_databases()
    config_path = kwargs.get("config_path", None)
    dictionary_path = kwargs.get("dictionary_path", None)
    source_path = kwargs["source_path"]
    output_model_path = kwargs["output_model_path"]

    if not source_path.lower().endswith(".arpa"):
        if not dictionary_path:
            trainer = MfaLmCorpusTrainer(
                corpus_directory=source_path,
                **MfaLmCorpusTrainer.parse_parameters(config_path, context.params, context.args),
            )
        else:
            trainer = MfaLmDictionaryCorpusTrainer(
                corpus_directory=source_path,
                dictionary_path=dictionary_path,
                **MfaLmDictionaryCorpusTrainer.parse_parameters(
                    config_path, context.params, context.args
                ),
            )
    else:
        trainer = MfaLmArpaTrainer(
            arpa_path=source_path,
            **MfaLmArpaTrainer.parse_parameters(config_path, context.params, context.args),
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
