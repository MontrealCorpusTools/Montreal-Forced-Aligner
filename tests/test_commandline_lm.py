import os

import click.testing

from montreal_forced_aligner.command_line.mfa import mfa_cli


def test_train_lm(
    basic_corpus_dir,
    temp_dir,
    generated_dir,
    basic_train_lm_config_path,
    db_setup,
):
    temp_dir = os.path.join(temp_dir, "train_lm")
    output_model_path = generated_dir.joinpath("test_basic_lm.zip")
    command = [
        "train_lm",
        basic_corpus_dir,
        output_model_path,
        "--config_path",
        basic_train_lm_config_path,
        "-q",
        "--clean",
    ]
    command = [str(x) for x in command]
    result = click.testing.CliRunner(mix_stderr=False).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value
    assert os.path.exists(output_model_path)


def test_train_lm_text(
    basic_split_dir,
    temp_dir,
    generated_dir,
    basic_train_lm_config_path,
    db_setup,
):
    temp_dir = os.path.join(temp_dir, "train_lm_text")
    text_dir = basic_split_dir[1]
    output_model_path = generated_dir.joinpath("test_basic_lm_split.zip")
    command = [
        "train_lm",
        text_dir,
        output_model_path,
        "--config_path",
        basic_train_lm_config_path,
        "-q",
        "--clean",
    ]
    command = [str(x) for x in command]
    click.testing.CliRunner().invoke(mfa_cli, command, catch_exceptions=False)
    assert os.path.exists(output_model_path)


def test_train_lm_dictionary(
    basic_split_dir,
    basic_dict_path,
    temp_dir,
    generated_dir,
    basic_train_lm_config_path,
    db_setup,
):
    temp_dir = os.path.join(temp_dir, "train_lm_dictionary")
    text_dir = basic_split_dir[1]
    output_model_path = generated_dir.joinpath("test_basic_lm_split.zip")
    command = [
        "train_lm",
        text_dir,
        output_model_path,
        "--dictionary_path",
        basic_dict_path,
        "--config_path",
        basic_train_lm_config_path,
        "-q",
        "--clean",
    ]
    command = [str(x) for x in command]
    click.testing.CliRunner().invoke(mfa_cli, command, catch_exceptions=False)
    assert os.path.exists(output_model_path)


def test_train_lm_arpa(
    transcription_language_model_arpa,
    temp_dir,
    generated_dir,
    basic_train_lm_config_path,
    db_setup,
):
    temp_dir = os.path.join(temp_dir, "train_lm_arpa")
    output_model_path = generated_dir.joinpath("test_basic_lm_split.zip")
    command = [
        "train_lm",
        transcription_language_model_arpa,
        output_model_path,
        "--config_path",
        basic_train_lm_config_path,
        "-q",
        "--clean",
    ]
    command = [str(x) for x in command]
    click.testing.CliRunner().invoke(mfa_cli, command, catch_exceptions=False)
    assert os.path.exists(output_model_path)


def test_train_lm_text_no_mp(
    basic_split_dir,
    temp_dir,
    generated_dir,
    basic_train_lm_config_path,
    db_setup,
):
    text_dir = basic_split_dir[1]
    output_model_path = generated_dir.joinpath("test_basic_lm_split.zip")
    command = [
        "train_lm",
        text_dir,
        output_model_path,
        "--config_path",
        basic_train_lm_config_path,
        "-q",
        "--clean",
        "-j",
        "1",
    ]
    command = [str(x) for x in command]
    click.testing.CliRunner().invoke(mfa_cli, command, catch_exceptions=False)
    assert os.path.exists(output_model_path)
