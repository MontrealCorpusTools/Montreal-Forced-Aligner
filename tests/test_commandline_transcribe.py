import os
import sys

import pytest

from montreal_forced_aligner.command_line.mfa import parser
from montreal_forced_aligner.command_line.transcribe import run_transcribe_corpus


def test_transcribe(
    basic_corpus_dir,
    sick_dict_path,
    english_acoustic_model,
    generated_dir,
    transcription_acoustic_model,
    transcription_language_model,
    temp_dir,
    transcribe_config_path,
):
    output_path = os.path.join(generated_dir, "transcribe_test")
    command = [
        "transcribe",
        basic_corpus_dir,
        sick_dict_path,
        transcription_acoustic_model,
        transcription_language_model,
        output_path,
        "-t",
        temp_dir,
        "-q",
        "--clean",
        "--debug",
        "-v",
        "--config_path",
        transcribe_config_path,
    ]
    args, unknown = parser.parse_known_args(command)
    run_transcribe_corpus(args)

    assert os.path.exists(os.path.join(output_path, "michael", "acoustic_corpus.lab"))


def test_transcribe_arpa(
    basic_corpus_dir,
    sick_dict_path,
    english_acoustic_model,
    generated_dir,
    transcription_language_model_arpa,
    temp_dir,
    transcribe_config_path,
):
    if sys.platform == "win32":
        pytest.skip("No LM generation on Windows")
    temp_dir = os.path.join(temp_dir, "arpa_test_temp")
    output_path = os.path.join(generated_dir, "transcribe_test_arpa")
    print(transcription_language_model_arpa)
    command = [
        "transcribe",
        basic_corpus_dir,
        sick_dict_path,
        english_acoustic_model,
        transcription_language_model_arpa,
        output_path,
        "-t",
        temp_dir,
        "-q",
        "--clean",
        "--debug",
        "-v",
        "--disable_mp",
        "false",
        "--config_path",
        transcribe_config_path,
    ]
    args, unknown = parser.parse_known_args(command)
    run_transcribe_corpus(args)
    assert os.path.exists(os.path.join(output_path, "michael", "acoustic_corpus.lab"))


def test_transcribe_speaker_dictionaries(
    multilingual_ipa_corpus_dir,
    ipa_speaker_dict_path,
    english_ipa_acoustic_model,
    generated_dir,
    transcription_language_model,
    temp_dir,
    transcribe_config_path,
):
    output_path = os.path.join(generated_dir, "transcribe_test")
    command = [
        "transcribe",
        multilingual_ipa_corpus_dir,
        ipa_speaker_dict_path,
        english_ipa_acoustic_model,
        transcription_language_model,
        output_path,
        "-t",
        temp_dir,
        "-q",
        "--clean",
        "--debug",
        "--config_path",
        transcribe_config_path,
    ]
    args, unknown = parser.parse_known_args(command)
    run_transcribe_corpus(args)


def test_transcribe_speaker_dictionaries_evaluate(
    multilingual_ipa_tg_corpus_dir,
    ipa_speaker_dict_path,
    english_ipa_acoustic_model,
    generated_dir,
    transcription_language_model,
    temp_dir,
    transcribe_config_path,
):
    output_path = os.path.join(generated_dir, "transcribe_test")
    command = [
        "transcribe",
        multilingual_ipa_tg_corpus_dir,
        ipa_speaker_dict_path,
        english_ipa_acoustic_model,
        transcription_language_model,
        output_path,
        "-t",
        temp_dir,
        "-q",
        "--clean",
        "--debug",
        "--config_path",
        transcribe_config_path,
        "--evaluate",
    ]
    args, unknown = parser.parse_known_args(command)
    run_transcribe_corpus(args)
