import os

from montreal_forced_aligner.command_line.mfa import parser
from montreal_forced_aligner.command_line.train_acoustic_model import run_train_acoustic_model


def test_train_and_align_basic(
    basic_corpus_dir,
    sick_dict_path,
    generated_dir,
    temp_dir,
    mono_train_config_path,
    textgrid_output_model_path,
):
    if os.path.exists(textgrid_output_model_path):
        os.remove(textgrid_output_model_path)
    command = [
        "train",
        basic_corpus_dir,
        sick_dict_path,
        os.path.join(generated_dir, "basic_output"),
        "-t",
        temp_dir,
        "--config_path",
        mono_train_config_path,
        "-q",
        "--clean",
        "--debug",
        "-o",
        textgrid_output_model_path,
    ]
    args, unknown = parser.parse_known_args(command)
    run_train_acoustic_model(args, unknown)
    assert os.path.exists(textgrid_output_model_path)


def test_train_and_align_basic_speaker_dict(
    multilingual_ipa_tg_corpus_dir,
    ipa_speaker_dict_path,
    generated_dir,
    temp_dir,
    basic_train_config_path,
    textgrid_output_model_path,
):
    if os.path.exists(textgrid_output_model_path):
        os.remove(textgrid_output_model_path)
    command = [
        "train",
        multilingual_ipa_tg_corpus_dir,
        ipa_speaker_dict_path,
        os.path.join(generated_dir, "ipa_speaker_output"),
        "-t",
        temp_dir,
        "--config_path",
        basic_train_config_path,
        "-q",
        "--clean",
        "--debug",
        "-o",
        textgrid_output_model_path,
    ]
    args, unknown = parser.parse_known_args(command)
    run_train_acoustic_model(args, unknown)
    assert os.path.exists(textgrid_output_model_path)
