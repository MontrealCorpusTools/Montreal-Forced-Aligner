"""Command line functions for aligning corpora"""
from __future__ import annotations

import os

import click
import yaml

from montreal_forced_aligner.alignment import PretrainedAligner
from montreal_forced_aligner.command_line.utils import (
    check_databases,
    cleanup_databases,
    common_options,
    validate_acoustic_model,
    validate_dictionary,
)
from montreal_forced_aligner.config import GLOBAL_CONFIG, MFA_PROFILE_VARIABLE
from montreal_forced_aligner.data import WorkflowType
from montreal_forced_aligner.helper import mfa_open

__all__ = ["align_corpus_cli"]


@click.command(
    name="align",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Align a corpus",
)
@click.argument("corpus_directory", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("dictionary_path", type=click.UNPROCESSED, callback=validate_dictionary)
@click.argument("acoustic_model_path", type=click.UNPROCESSED, callback=validate_acoustic_model)
@click.argument("output_directory", type=click.Path(file_okay=False, dir_okay=True))
@click.option(
    "--config_path",
    "-c",
    help="Path to config file to use for training.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
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
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--reference_directory",
    help="Directory containing gold standard alignments to evaluate",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--custom_mapping_path",
    help="YAML file for mapping phones across phone sets in evaluations.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--output_format",
    help="Format for aligned output files (default is long_textgrid).",
    default="long_textgrid",
    type=click.Choice(["long_textgrid", "short_textgrid", "json", "csv"]),
)
@click.option(
    "--include_original_text",
    is_flag=True,
    help="Flag to include original utterance text in the output.",
    default=False,
)
@click.option(
    "--fine_tune", is_flag=True, help="Flag for running extra fine tuning stage.", default=False
)
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def align_corpus_cli(context, **kwargs) -> None:
    """
    Align a corpus with a pronunciation dictionary and a pretrained acoustic model.
    """
    if kwargs.get("profile", None) is not None:
        os.putenv(MFA_PROFILE_VARIABLE, kwargs["profile"])
    GLOBAL_CONFIG.current_profile.update(kwargs)
    GLOBAL_CONFIG.save()
    check_databases()
    config_path = kwargs.get("config_path", None)
    reference_directory = kwargs.get("reference_directory", None)
    custom_mapping_path = kwargs.get("custom_mapping_path", None)
    corpus_directory = kwargs["corpus_directory"]
    dictionary_path = kwargs["dictionary_path"]
    acoustic_model_path = kwargs["acoustic_model_path"]
    output_directory = kwargs["output_directory"]
    output_format = kwargs["output_format"]
    include_original_text = kwargs["include_original_text"]
    aligner = PretrainedAligner(
        corpus_directory=corpus_directory,
        dictionary_path=dictionary_path,
        acoustic_model_path=acoustic_model_path,
        **PretrainedAligner.parse_parameters(config_path, context.params, context.args),
    )
    try:
        aligner.align()
        if aligner.use_phone_model:
            aligner.export_files(
                output_directory,
                output_format=output_format,
                include_original_text=include_original_text,
            )
        else:
            aligner.export_files(
                output_directory,
                output_format=output_format,
                include_original_text=include_original_text,
            )
        if reference_directory:
            mapping = None
            if custom_mapping_path:
                with mfa_open(custom_mapping_path, "r") as f:
                    mapping = yaml.safe_load(f)
            aligner.load_reference_alignments(reference_directory)
            reference_alignments = WorkflowType.reference
        else:
            reference_alignments = WorkflowType.alignment

        if aligner.use_phone_model:
            aligner.evaluate_alignments(
                mapping,
                output_directory=output_directory,
                reference_source=reference_alignments,
                comparison_source=WorkflowType.phone_transcription,
            )
        else:
            if reference_alignments is WorkflowType.reference:
                aligner.evaluate_alignments(
                    mapping,
                    output_directory=output_directory,
                    reference_source=reference_alignments,
                    comparison_source=WorkflowType.alignment,
                )
    except Exception:
        aligner.dirty = True
        raise
    finally:
        aligner.cleanup()
        cleanup_databases()
