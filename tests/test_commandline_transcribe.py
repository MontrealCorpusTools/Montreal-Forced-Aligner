import os

import click.testing
import pytest

from montreal_forced_aligner.command_line.mfa import mfa_cli
from montreal_forced_aligner.transcription.models import FOUND_WHISPERX
from montreal_forced_aligner.transcription.multiprocessing import FOUND_SPEECHBRAIN


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
        "--no_use_mp",
        "-v",
        "--config_path",
        transcribe_config_path,
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

    assert os.path.exists(os.path.join(output_path, "michael", "acoustic_corpus.lab"))


def test_transcribe_speechbrain(
    combined_corpus_dir,
    generated_dir,
    transcription_acoustic_model,
    transcription_language_model,
    temp_dir,
    db_setup,
):
    if not FOUND_SPEECHBRAIN:
        pytest.skip("SpeechBrain not installed")
    output_path = generated_dir.joinpath("transcribe_test_sb")
    command = [
        "transcribe_speechbrain",
        combined_corpus_dir,
        "english",
        output_path,
        "--architecture",
        "wav2vec2",
        "--clean",
        "--no_debug",
        "--evaluate",
        "--no_cuda",
        "--use_postgres",
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
    assert os.path.exists(output_path)


def test_transcribe_whisper(
    combined_corpus_dir,
    generated_dir,
    transcription_acoustic_model,
    transcription_language_model,
    temp_dir,
    db_setup,
):
    if not FOUND_WHISPERX:
        pytest.skip("whisperx not installed")
    output_path = generated_dir.joinpath("transcribe_test_whisper")
    command = [
        "transcribe_whisper",
        combined_corpus_dir,
        output_path,
        "--language",
        "english",
        "--architecture",
        "tiny",
        "--clean",
        "--no_debug",
        "--evaluate",
        "--cuda",
        "--use_postgres",
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
    assert os.path.exists(output_path)


def test_transcribe_arpa(
    basic_corpus_dir,
    english_dictionary,
    english_acoustic_model,
    generated_dir,
    transcription_language_model_arpa,
    temp_dir,
    transcribe_config_path,
    db_setup,
):
    output_path = generated_dir.joinpath("transcribe_test_arpa")
    command = [
        "transcribe",
        basic_corpus_dir,
        english_dictionary,
        english_acoustic_model,
        transcription_language_model_arpa,
        output_path,
        "-q",
        "--clean",
        "--no_debug",
        "-v",
        "--use_mp",
        "--config_path",
        transcribe_config_path,
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
    result = click.testing.CliRunner(mix_stderr=False).invoke(
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
    result = click.testing.CliRunner(mix_stderr=False).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value

    assert os.path.exists(output_path)
