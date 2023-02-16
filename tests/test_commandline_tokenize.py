import os

import click.testing

from montreal_forced_aligner.command_line.mfa import mfa_cli


def test_tokenize_pretrained(japanese_tokenizer_model, japanese_dir, temp_dir, generated_dir):
    out_directory = generated_dir.joinpath("japanese_tokenized")
    command = [
        "tokenize",
        japanese_dir,
        japanese_tokenizer_model,
        out_directory,
        "-t",
        os.path.join(temp_dir, "tokenize_cli"),
        "-q",
        "--clean",
        "--use_mp",
        "False",
    ]
    command = [str(x) for x in command]
    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value
    assert os.path.exists(out_directory)


def test_train_tokenizer(combined_corpus_dir, temp_dir, generated_dir):
    output_path = generated_dir.joinpath("test_tokenizer.zip")
    command = [
        "train_tokenizer",
        combined_corpus_dir,
        output_path,
        "-t",
        os.path.join(temp_dir, "test_train_tokenizer"),
        "-q",
        "--clean",
        "--debug",
        "--validate",
    ]
    command = [str(x) for x in command]
    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value
    assert os.path.exists(output_path)


def test_train_tokenizer_phonetisaurus(combined_corpus_dir, temp_dir, generated_dir):
    output_path = generated_dir.joinpath("test_tokenizer_model_phonetisaurus.zip")
    command = [
        "train_tokenizer",
        combined_corpus_dir,
        output_path,
        "-t",
        os.path.join(temp_dir, "test_train_tokenizer_phonetisaurus"),
        "-q",
        "--clean",
        "--debug",
        "--phonetisaurus",
        "--validate",
    ]
    command = [str(x) for x in command]
    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value
    assert os.path.exists(output_path)


def test_tokenize_textgrid(
    multilingual_ipa_tg_corpus_dir,
    test_tokenizer_model,
    generated_dir,
    temp_dir,
    g2p_config_path,
):
    output_directory = generated_dir.joinpath("tokenized_tg")
    command = [
        "tokenize",
        multilingual_ipa_tg_corpus_dir,
        test_tokenizer_model,
        output_directory,
        "-t",
        os.path.join(temp_dir, "tokenizer_cli"),
        "-q",
        "--clean",
        "--debug",
    ]
    command = [str(x) for x in command]
    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value
    assert os.path.exists(output_directory)


def test_tokenize_textgrid_phonetisaurus(
    multilingual_ipa_tg_corpus_dir,
    test_tokenizer_model_phonetisaurus,
    generated_dir,
    temp_dir,
    g2p_config_path,
):
    output_directory = generated_dir.joinpath("tokenized_tg")
    command = [
        "tokenize",
        multilingual_ipa_tg_corpus_dir,
        test_tokenizer_model_phonetisaurus,
        output_directory,
        "-t",
        os.path.join(temp_dir, "tokenizer_cli"),
        "-q",
        "--clean",
        "--debug",
    ]
    command = [str(x) for x in command]
    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value
    assert os.path.exists(output_directory)
