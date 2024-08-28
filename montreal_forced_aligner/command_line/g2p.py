"""Command line functions for generating pronunciations using G2P models"""
from __future__ import annotations

import pathlib
import sys
from pathlib import Path

import rich_click as click

from montreal_forced_aligner import config
from montreal_forced_aligner.command_line.utils import (
    common_options,
    validate_dictionary,
    validate_g2p_model,
)
from montreal_forced_aligner.g2p.generator import (
    PyniniConsoleGenerator,
    PyniniCorpusGenerator,
    PyniniDictionaryCorpusGenerator,
    PyniniWordListGenerator,
)

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
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path, allow_dash=True),
)
@click.argument("g2p_model_path", type=click.UNPROCESSED, callback=validate_g2p_model)
@click.argument(
    "output_path", type=click.Path(file_okay=True, dir_okay=True, path_type=Path, allow_dash=True)
)
@click.option(
    "--config_path",
    "-c",
    help="Path to config file to use for G2P.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--num_pronunciations",
    "-n",
    help="Number of pronunciations to generate.",
    type=click.INT,
    default=0,
)
@click.option(
    "--dictionary_path",
    help="Path to existing pronunciation dictionary to use to find OOVs.",
    type=click.UNPROCESSED,
    callback=validate_dictionary,
)
@click.option(
    "--include_bracketed",
    is_flag=True,
    help="Included words enclosed by brackets, job_name.e. [...], (...), <...>.",
    default=False,
)
@click.option(
    "--export_scores",
    is_flag=True,
    help="Add a column to export for the score of the generated pronunciation.",
    default=False,
)
@click.option(
    "--sorted",
    is_flag=True,
    help="Ensure output file is sorted alphabetically (slower).",
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
        config.profile = kwargs.pop("profile")
    config.update_configuration(kwargs)

    config_path = kwargs.get("config_path", None)
    input_path = kwargs["input_path"]
    g2p_model_path = kwargs["g2p_model_path"]
    output_path = kwargs["output_path"]
    dictionary_path = kwargs.get("dictionary_path", None)
    use_stdin = input_path == pathlib.Path("-")
    use_stdout = output_path == pathlib.Path("-")
    export_scores = kwargs.get("export_scores", False)

    if input_path.is_dir():
        per_utterance = False
        if not output_path.suffix:
            per_utterance = True
        if dictionary_path is not None:
            g2p = PyniniDictionaryCorpusGenerator(
                corpus_directory=input_path,
                dictionary_path=dictionary_path,
                g2p_model_path=g2p_model_path,
                **PyniniDictionaryCorpusGenerator.parse_parameters(
                    config_path, context.params, context.args
                ),
            )
        else:
            g2p = PyniniCorpusGenerator(
                corpus_directory=input_path,
                g2p_model_path=g2p_model_path,
                per_utterance=per_utterance,
                **PyniniCorpusGenerator.parse_parameters(
                    config_path, context.params, context.args
                ),
            )
            if per_utterance:
                g2p.num_pronunciations = 1
    elif use_stdin:
        g2p = PyniniConsoleGenerator(
            g2p_model_path=g2p_model_path,
            **PyniniWordListGenerator.parse_parameters(config_path, context.params, context.args),
        )
    else:
        g2p = PyniniWordListGenerator(
            word_list_path=input_path,
            g2p_model_path=g2p_model_path,
            **PyniniWordListGenerator.parse_parameters(config_path, context.params, context.args),
        )

    try:
        g2p.setup()
        if use_stdin:
            if use_stdout:
                output = sys.stdout
            else:
                output = open(output_path, "w", encoding="utf8")
            try:
                for line in sys.stdin:
                    word = line.strip().lower()
                    if not word:
                        continue
                    pronunciations = g2p.rewriter(word)
                    if not pronunciations:
                        if export_scores:
                            output.write(f"{word}\t\t\n")
                        else:
                            output.write(f"{word}\t\n")
                    for p, score in pronunciations:
                        if export_scores:
                            output.write(f"{word}\t{p}\t{score}\n")
                        else:
                            output.write(f"{word}\t{p}\n")
                    output.flush()
            finally:
                output.close()
        else:
            g2p.export_pronunciations(
                output_path, export_scores=export_scores, ensure_sorted=kwargs.get("sorted", False)
            )
    except Exception:
        g2p.dirty = True
        raise
    finally:
        g2p.cleanup()
