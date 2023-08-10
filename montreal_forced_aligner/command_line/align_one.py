"""Command line functions for aligning single files"""
from __future__ import annotations

import os
from pathlib import Path

import rich_click as click
import yaml

from montreal_forced_aligner.alignment import PretrainedAligner
from montreal_forced_aligner.command_line.utils import (
    common_options,
    validate_acoustic_model,
    validate_dictionary,
)
from montreal_forced_aligner.config import GLOBAL_CONFIG, MFA_PROFILE_VARIABLE
from montreal_forced_aligner.data import WorkflowType
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.models import AcousticModel

from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.feat.mfcc import MfccComputer

__all__ = ["align_one_cli"]


@click.command(
    name="align_one",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Align a single file",
)
@click.argument(
    "sound_file_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "text_file_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.argument("dictionary_path", type=click.UNPROCESSED, callback=validate_dictionary)
@click.argument("acoustic_model_path", type=click.UNPROCESSED, callback=validate_acoustic_model)
@click.argument(
    "output_path", type=click.Path(file_okay=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--config_path",
    "-c",
    help="Path to config file to use for aligning.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
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
def align_one_cli(context, **kwargs) -> None:
    """
    Align a single file with a pronunciation dictionary and a pretrained acoustic model.
    """
    if kwargs.get("profile", None) is not None:
        os.environ[MFA_PROFILE_VARIABLE] = kwargs.pop("profile")
    GLOBAL_CONFIG.current_profile.update(kwargs)
    GLOBAL_CONFIG.save()
    config_path = kwargs.get("config_path", None)
    sound_file_path = kwargs["sound_file_path"]
    dictionary_path: Path = kwargs["dictionary_path"]
    acoustic_model_path = kwargs["acoustic_model_path"]
    output_path = kwargs["output_path"]
    output_format = kwargs["output_format"]
    include_original_text = kwargs["include_original_text"]
    config = PretrainedAligner.parse_parameters(config_path, context.params, context.args)

    extracted_models_dir = GLOBAL_CONFIG.current_profile.temporary_directory.joinpath("extracted_models", "dictionary")
    dictionary_path = extracted_models_dir.joinpath(dictionary_path.stem)
    dictionary_path.mkdir(parents=True, exist_ok=True)
    lc = LexiconCompiler(
        silence_disambiguation_symbol=config['silence_disambiguation_symbol'],
        silence_probability=config['silence_probability'],
        initial_silence_probability=config['initial_silence_probability'],
        final_silence_correction=config['final_silence_correction'],
        final_non_silence_correction=config['final_non_silence_correction'],
        silence_word=config['silence_word'],
        oov_word=config['oov_word'],
        silence_phone=config['optional_silence_phone'],
        oov_phone=config['oov_phone'],
        position_dependent_phones=config['position_dependent_phones'],
        ignore_case=config['ignore_case'],
    )
    l_fst_path = dictionary_path.joinpath('L.fst')
    acoustic_model = AcousticModel(acoustic_model_path)
    if l_fst_path.exists():
        lc.load_fst(l_fst_path)
    else:
        lc.load_pronunciations(dictionary_path)
        lc.fst.write(str(l_fst_path))
        lc.clear()
    mfcc_computer = MfccComputer(**acoustic_model.mfcc_options)

