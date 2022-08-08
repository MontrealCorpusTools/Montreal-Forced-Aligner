import os

from montreal_forced_aligner.command_line.mfa import parser
from montreal_forced_aligner.command_line.train_acoustic_model import run_train_acoustic_model


def test_train_acoustic_with_g2p(
    basic_corpus_dir,
    english_us_mfa_dictionary,
    generated_dir,
    temp_dir,
    train_g2p_acoustic_config_path,
    acoustic_g2p_model_path,
):
    if os.path.exists(acoustic_g2p_model_path):
        os.remove(acoustic_g2p_model_path)
    command = [
        "train",
        basic_corpus_dir,
        english_us_mfa_dictionary,
        os.path.join(generated_dir, "basic_output"),
        "-t",
        temp_dir,
        "-q",
        "--clean",
        "--quiet",
        "--debug",
        "--config_path",
        train_g2p_acoustic_config_path,
        "-o",
        acoustic_g2p_model_path,
    ]
    args, unknown = parser.parse_known_args(command)
    run_train_acoustic_model(args, unknown)
    assert os.path.exists(acoustic_g2p_model_path)


def test_train_and_align_basic_speaker_dict(
    multilingual_ipa_tg_corpus_dir,
    mfa_speaker_dict_path,
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
        mfa_speaker_dict_path,
        os.path.join(generated_dir, "ipa speaker output"),
        "-t",
        os.path.join(temp_dir, "temp dir with spaces"),
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
