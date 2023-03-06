import os

import click.testing

from montreal_forced_aligner.command_line.mfa import mfa_cli


def test_adapt_basic(
    basic_corpus_dir,
    generated_dir,
    english_dictionary,
    temp_dir,
    test_align_config,
    english_acoustic_model,
    db_setup,
):
    adapted_model_path = generated_dir.joinpath("basic_adapted.zip")
    command = [
        "adapt",
        basic_corpus_dir,
        english_dictionary,
        english_acoustic_model,
        adapted_model_path,
        "--beam",
        "15",
        "--clean",
        "--no_debug",
        "-p",
        "test",
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
    assert os.path.exists(adapted_model_path)


def test_adapt_multilingual(
    multilingual_ipa_corpus_dir,
    mfa_speaker_dict_path,
    generated_dir,
    temp_dir,
    basic_align_config_path,
    english_acoustic_model,
    english_mfa_acoustic_model,
    db_setup,
):
    adapted_model_path = generated_dir.joinpath("multilingual_adapted.zip")
    output_path = generated_dir.joinpath("multilingual_output")
    command = [
        "adapt",
        multilingual_ipa_corpus_dir,
        mfa_speaker_dict_path,
        english_mfa_acoustic_model,
        adapted_model_path,
        output_path,
        "--config_path",
        basic_align_config_path,
        "-q",
        "--clean",
        "--no_debug",
        "-p",
        "test",
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
    assert os.path.exists(adapted_model_path)
