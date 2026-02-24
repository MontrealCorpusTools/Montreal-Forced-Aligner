"""Command line functions for aligning corpora"""
from __future__ import annotations

import typing
from pathlib import Path

import rich_click as click

from montreal_forced_aligner.command_line.utils import (
    common_options,
    initialize_configuration,
    validate_corpus_directory,
)
from montreal_forced_aligner.corpus.alignment_comparer import (
    AlignmentAudioComparer,
    AlignmentComparer,
)
from montreal_forced_aligner.data import WorkflowType

__all__ = ["compare_alignments_cli"]


@click.command(
    name="compare_alignments",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Compare two sets of alignments",
)
@click.argument("reference_directory", type=click.UNPROCESSED, callback=validate_corpus_directory)
@click.argument("test_directory", type=click.UNPROCESSED, callback=validate_corpus_directory)
@click.argument(
    "output_directory", type=click.Path(file_okay=False, dir_okay=True, path_type=Path)
)
@click.option(
    "--custom_mapping_path",
    help="YAML file for mapping phones from acoustic model phone set to phone set in golden alignments.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--config_path",
    "-c",
    help="Path to config file to use for aligning.",
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
    "--strict_mapping",
    is_flag=True,
    help="Flag for ensuring all phones are in the given mapping file.",
    default=False,
)
@click.option(
    "--naive",
    is_flag=True,
    help="Flag for skipping interval alignment and using closest boundary for error calculation.",
    default=False,
)
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def compare_alignments_cli(context, **kwargs) -> None:
    """
    Align a corpus with a pronunciation dictionary and a pretrained acoustic model.
    """
    initialize_configuration(context)
    custom_mapping_path: typing.Optional[Path] = kwargs.get("custom_mapping_path", None)
    reference_directory = kwargs["reference_directory"].absolute()
    test_directory = kwargs["test_directory"].absolute()
    output_directory = kwargs["output_directory"]
    audio_directory = kwargs.get("audio_directory", None)
    strict_mapping = kwargs.get("strict_mapping", False)
    naive = kwargs.get("naive", False)
    if audio_directory is None:
        alignment_comparer = AlignmentComparer(
            corpus_directory=reference_directory,
            test_directory=test_directory,
        )
    else:
        alignment_comparer = AlignmentAudioComparer(
            corpus_directory=reference_directory,
            test_directory=test_directory,
            audio_directory=audio_directory,
        )
    try:
        alignment_comparer.load_corpus()

        if custom_mapping_path is not None:
            alignment_comparer.load_mapping(custom_mapping_path, strict=strict_mapping)
        alignment_comparer.evaluate_alignments(
            output_directory=output_directory,
            reference_source=WorkflowType.reference,
            comparison_source=WorkflowType.alignment,
            naive=naive,
        )
    except Exception:
        alignment_comparer.dirty = True
        raise
    finally:
        alignment_comparer.cleanup()
