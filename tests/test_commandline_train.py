import os

import click.testing

from montreal_forced_aligner.command_line.mfa import mfa_cli


def test_train_acoustic_with_g2p(
    combined_corpus_dir,
    english_us_mfa_dictionary,
    generated_dir,
    temp_dir,
    train_g2p_acoustic_config_path,
    acoustic_g2p_model_path,
):
    if os.path.exists(acoustic_g2p_model_path):
        os.remove(acoustic_g2p_model_path)
    output_directory = os.path.join(generated_dir, "train_g2p_textgrids")
    command = [
        "train",
        combined_corpus_dir,
        english_us_mfa_dictionary,
        acoustic_g2p_model_path,
        "--output_directory",
        output_directory,
        "-t",
        os.path.join(temp_dir, "train_cli"),
        "-q",
        "--clean",
        "--quiet",
        "--debug",
        "--config_path",
        train_g2p_acoustic_config_path,
    ]
    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value
    assert os.path.exists(acoustic_g2p_model_path)
    assert os.path.exists(output_directory)


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
    output_directory = os.path.join(generated_dir, "ipa speaker output")
    command = [
        "train",
        multilingual_ipa_tg_corpus_dir,
        mfa_speaker_dict_path,
        textgrid_output_model_path,
        "-t",
        os.path.join(temp_dir, "temp dir with spaces"),
        "--config_path",
        basic_train_config_path,
        "-q",
        "--clean",
        "--no_debug",
        "--output_directory",
        output_directory,
        "--single_speaker",
    ]
    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value
    assert os.path.exists(textgrid_output_model_path)
    assert os.path.exists(output_directory)
