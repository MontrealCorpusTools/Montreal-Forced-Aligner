"""Command line functions for generating pronunciations using G2P models"""
from __future__ import annotations

import os
from pathlib import Path

import click

from montreal_forced_aligner.command_line.utils import (
    check_databases,
    cleanup_databases,
    common_options,
    validate_g2p_model,
)
from montreal_forced_aligner.config import GLOBAL_CONFIG, MFA_PROFILE_VARIABLE
from montreal_forced_aligner.g2p.generator import PyniniCorpusGenerator, PyniniWordListGenerator

__all__ = ["g2p_cli"]


@click.command(
    name="g2p",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Generate pronunciations",
)
@click.argument(
    "input_path", type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path)
)
@click.argument("g2p_model_path", type=click.UNPROCESSED, callback=validate_g2p_model)
@click.argument("output_path", type=click.Path(file_okay=True, dir_okay=False, path_type=Path))
@click.option(
    "--config_path",
    "-c",
    help="Path to config file to use for training.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--include_bracketed",
    is_flag=True,
    help="Included words enclosed by brackets, job_name.e. [...], (...), <...>.",
    default=False,
)
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def g2p_cli(context, **kwargs) -> None:
    """
    Generate a pronunciation dictionary using a G2P model.
    """
    if kwargs.get("profile", None) is not None:
        os.putenv(MFA_PROFILE_VARIABLE, kwargs["profile"])
    GLOBAL_CONFIG.current_profile.update(kwargs)
    GLOBAL_CONFIG.save()
    check_databases()

    config_path = kwargs.get("config_path", None)
    input_path = kwargs["input_path"]
    g2p_model_path = kwargs["g2p_model_path"]
    output_path = kwargs["output_path"]

    if os.path.isdir(input_path):
        g2p = PyniniCorpusGenerator(
            corpus_directory=input_path,
            g2p_model_path=g2p_model_path,
            **PyniniCorpusGenerator.parse_parameters(config_path, context.params, context.args),
        )
    else:
        g2p = PyniniWordListGenerator(
            word_list_path=input_path,
            g2p_model_path=g2p_model_path,
            **PyniniWordListGenerator.parse_parameters(config_path, context.params, context.args),
        )

    try:
        g2p.setup()
        g2p.export_pronunciations(output_path)
    except Exception:
        g2p.dirty = True
        raise
    finally:
        g2p.cleanup()
        cleanup_databases()
