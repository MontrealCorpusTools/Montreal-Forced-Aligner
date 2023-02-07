import os

import click.testing

from montreal_forced_aligner.command_line.mfa import mfa_cli


def test_validate_corpus(
    multilingual_ipa_tg_corpus_dir,
    english_mfa_acoustic_model,
    english_us_mfa_dictionary,
    temp_dir,
):
    command = [
        "validate",
        multilingual_ipa_tg_corpus_dir,
        english_us_mfa_dictionary,
        "--acoustic_model_path",
        english_mfa_acoustic_model,
        "-t",
        os.path.join(temp_dir, "validate_cli"),
        "-q",
        "--oov_count_threshold",
        "0",
        "--clean",
        "--no_use_mp",
        "--test_transcriptions",
        "--phone_confidence",
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


def test_validate_training_corpus(
    multilingual_ipa_tg_corpus_dir,
    english_dictionary,
    temp_dir,
    mono_train_config_path,
):

    command = [
        "validate",
        multilingual_ipa_tg_corpus_dir,
        english_dictionary,
        "-t",
        os.path.join(temp_dir, "validation"),
        "-q",
        "--clean",
        "--no_debug",
        "--config_path",
        mono_train_config_path,
        "--test_transcriptions",
        "--phone_confidence",
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


def test_validate_xsampa(
    xsampa_corpus_dir,
    xsampa_dict_path,
    temp_dir,
    xsampa_train_config_path,
):

    command = [
        "validate",
        xsampa_corpus_dir,
        xsampa_dict_path,
        "-t",
        os.path.join(temp_dir, "validation_xsampa"),
        "-q",
        "--clean",
        "--ignore_acoustics",
        "--config_path",
        xsampa_train_config_path,
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


def test_validate_dictionary(
    english_us_mfa_g2p_model,
    english_us_mfa_dictionary_subset,
    temp_dir,
):

    command = [
        "validate_dictionary",
        english_us_mfa_dictionary_subset,
        "--g2p_model_path",
        english_us_mfa_g2p_model,
        "-t",
        os.path.join(temp_dir, "dictionary_validation"),
        "-j",
        "1",
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


def test_validate_dictionary_train(
    basic_dict_path,
    temp_dir,
):

    command = [
        "validate_dictionary",
        basic_dict_path,
        "-t",
        os.path.join(temp_dir, "dictionary_validation_train"),
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
