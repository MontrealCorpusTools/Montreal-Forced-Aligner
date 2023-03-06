import os

import click.testing

from montreal_forced_aligner.command_line.mfa import mfa_cli


def test_transcribe(
    basic_corpus_dir,
    basic_dict_path,
    english_acoustic_model,
    generated_dir,
    transcription_acoustic_model,
    transcription_language_model,
    temp_dir,
    transcribe_config_path,
    db_setup,
):
    output_path = generated_dir.joinpath("transcribe_test")
    command = [
        "transcribe",
        basic_corpus_dir,
        basic_dict_path,
        transcription_acoustic_model,
        transcription_language_model,
        output_path,
        "-q",
        "--clean",
        "--debug",
        "-v",
        "--config_path",
        transcribe_config_path,
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

    assert os.path.exists(os.path.join(output_path, "michael", "acoustic_corpus.lab"))


def test_transcribe_arpa(
    basic_corpus_dir,
    basic_dict_path,
    english_acoustic_model,
    generated_dir,
    transcription_language_model_arpa,
    temp_dir,
    transcribe_config_path,
    db_setup,
):
    temp_dir = os.path.join(temp_dir, "arpa_test_temp")
    output_path = generated_dir.joinpath("transcribe_test_arpa")
    command = [
        "transcribe",
        basic_corpus_dir,
        basic_dict_path,
        english_acoustic_model,
        transcription_language_model_arpa,
        output_path,
        "-q",
        "--clean",
        "--no_debug",
        "-v",
        "--use_mp",
        "false",
        "--config_path",
        transcribe_config_path,
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
    assert os.path.exists(os.path.join(output_path, "michael", "acoustic_corpus.lab"))


def test_transcribe_speaker_dictionaries(
    multilingual_ipa_corpus_dir,
    mfa_speaker_dict_path,
    english_mfa_acoustic_model,
    generated_dir,
    transcription_language_model,
    temp_dir,
    transcribe_config_path,
    db_setup,
):
    output_path = generated_dir.joinpath("transcribe_test")
    command = [
        "transcribe",
        multilingual_ipa_corpus_dir,
        mfa_speaker_dict_path,
        english_mfa_acoustic_model,
        transcription_language_model,
        output_path,
        "-q",
        "--clean",
        "--debug",
        "--config_path",
        transcribe_config_path,
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


def test_transcribe_speaker_dictionaries_evaluate(
    multilingual_ipa_tg_corpus_dir,
    mfa_speaker_dict_path,
    english_mfa_acoustic_model,
    generated_dir,
    transcription_language_model,
    temp_dir,
    transcribe_config_path,
    db_setup,
):
    output_path = generated_dir.joinpath("transcribe_test")
    command = [
        "transcribe",
        multilingual_ipa_tg_corpus_dir,
        mfa_speaker_dict_path,
        english_mfa_acoustic_model,
        transcription_language_model,
        output_path,
        "-q",
        "--clean",
        "--debug",
        "--no_use_mp",
        "--language_model_weight",
        "16",
        "--word_insertion_penalty",
        "1.0",
        "--config_path",
        transcribe_config_path,
        "--evaluate",
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
