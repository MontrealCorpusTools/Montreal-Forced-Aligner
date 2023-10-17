"""Command line functions for training ivector extractors"""
from __future__ import annotations

from pathlib import Path

import rich_click as click

from montreal_forced_aligner import config
from montreal_forced_aligner.command_line.utils import common_options
from montreal_forced_aligner.ivector.trainer import TrainableIvectorExtractor

__all__ = ["train_ivector_cli"]


@click.command(
    name="train_ivector",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Train an ivector extractor",
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
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def train_ivector_cli(context, **kwargs) -> None:
    """
    Train an ivector extractor from a corpus and pretrained acoustic model.
    """
    if kwargs.get("profile", None) is not None:
        config.profile = kwargs.pop("profile")
    config.update_configuration(kwargs)
    config_path = kwargs.get("config_path", None)
    corpus_directory = kwargs["corpus_directory"].absolute()
    output_model_path = kwargs["output_model_path"]

    trainer = TrainableIvectorExtractor(
        corpus_directory=corpus_directory,
        **TrainableIvectorExtractor.parse_parameters(config_path, context.params, context.args),
    )

    try:

        trainer.train()
        trainer.export_model(output_model_path)

    except Exception:
        trainer.dirty = True
        raise
    finally:
        trainer.cleanup()
