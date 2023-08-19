"""Command line functions for aligning single files"""
from __future__ import annotations

import os
from pathlib import Path

import pywrapfst
import rich_click as click
from kalpy.feat.cmvn import CmvnComputer
from kalpy.fstext.lexicon import HierarchicalCtm, LexiconCompiler
from kalpy.utterance import Segment
from kalpy.utterance import Utterance as KalpyUtterance

from montreal_forced_aligner.alignment import PretrainedAligner
from montreal_forced_aligner.command_line.utils import (
    common_options,
    validate_acoustic_model,
    validate_dictionary,
)
from montreal_forced_aligner.config import GLOBAL_CONFIG, MFA_PROFILE_VARIABLE
from montreal_forced_aligner.corpus.classes import FileData
from montreal_forced_aligner.models import AcousticModel
from montreal_forced_aligner.online.alignment import align_utterance_online

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
@click.argument("output_path", type=click.Path(file_okay=True, dir_okay=False, path_type=Path))
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
    config_path = kwargs.get("config_path", None)
    sound_file_path = kwargs["sound_file_path"]
    text_file_path = kwargs["text_file_path"]
    dictionary_path: Path = kwargs["dictionary_path"]
    acoustic_model_path = kwargs["acoustic_model_path"]
    output_path: Path = kwargs["output_path"]
    output_format = kwargs["output_format"]
    config = PretrainedAligner.parse_parameters(config_path, context.params, context.args)

    acoustic_model = AcousticModel(acoustic_model_path)
    extracted_models_dir = GLOBAL_CONFIG.current_profile.temporary_directory.joinpath(
        "extracted_models", "dictionary"
    )
    dictionary_directory = extracted_models_dir.joinpath(dictionary_path.stem)
    dictionary_directory.mkdir(parents=True, exist_ok=True)
    lexicon_compiler = LexiconCompiler(
        disambiguation=False,
        silence_probability=acoustic_model.parameters["silence_probability"],
        initial_silence_probability=acoustic_model.parameters["initial_silence_probability"],
        final_silence_correction=acoustic_model.parameters["final_silence_correction"],
        final_non_silence_correction=acoustic_model.parameters["final_non_silence_correction"],
        silence_phone=acoustic_model.parameters["optional_silence_phone"],
        oov_phone=acoustic_model.parameters["oov_phone"],
        position_dependent_phones=acoustic_model.parameters["position_dependent_phones"],
        phones=acoustic_model.parameters["non_silence_phones"],
        ignore_case=config.get("ignore_case", True),
    )
    l_fst_path = dictionary_directory.joinpath("L.fst")
    l_align_fst_path = dictionary_directory.joinpath("L_align.fst")
    words_path = dictionary_directory.joinpath("words.txt")
    phones_path = dictionary_directory.joinpath("phones.txt")
    if l_fst_path.exists():
        lexicon_compiler.load_l_from_file(l_fst_path)
        lexicon_compiler.load_l_align_from_file(l_align_fst_path)
        lexicon_compiler.word_table = pywrapfst.SymbolTable.read_text(words_path)
        lexicon_compiler.phone_table = pywrapfst.SymbolTable.read_text(phones_path)
    else:
        lexicon_compiler.load_pronunciations(dictionary_path)
        lexicon_compiler.fst.write(str(l_fst_path))
        lexicon_compiler.align_fst.write(str(l_align_fst_path))
        lexicon_compiler.word_table.write_text(words_path)
        lexicon_compiler.phone_table.write_text(phones_path)
        lexicon_compiler.clear()
    file_name = sound_file_path.stem
    file = FileData.parse_file(file_name, sound_file_path, text_file_path, "", 0)
    file_ctm = HierarchicalCtm([])
    utterances = []
    cmvn_computer = CmvnComputer()
    for utterance in file.utterances:
        seg = Segment(sound_file_path, utterance.begin, utterance.end, utterance.channel)
        utt = KalpyUtterance(seg, utterance.text)
        utt.generate_mfccs(acoustic_model.mfcc_computer)
        utterances.append(utt)
    cmvn = cmvn_computer.compute_cmvn_from_features([utt.mfccs for utt in utterances])
    align_options = {
        k: v
        for k, v in config
        if k
        in [
            "beam",
            "retry_beam",
            "acoustic_scale",
            "transition_scale",
            "self_loop_scale",
            "boost_silence",
        ]
    }
    for utt in utterances:
        utt.apply_cmvn(cmvn)
        ctm = align_utterance_online(acoustic_model, utt, lexicon_compiler, **align_options)
        file_ctm.word_intervals.extend(ctm.word_intervals)
    if str(output_path) != "-":

        output_path.parent.mkdir(parents=True, exist_ok=True)
    file_ctm.export_textgrid(
        output_path, file_duration=file.wav_info.duration, output_format=output_format
    )
