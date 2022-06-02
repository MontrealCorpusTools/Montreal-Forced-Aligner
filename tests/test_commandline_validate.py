import os
import sys

import pytest

from montreal_forced_aligner.command_line.mfa import parser
from montreal_forced_aligner.command_line.validate import (
    run_validate_corpus,
    run_validate_dictionary,
)
from montreal_forced_aligner.exceptions import ArgumentError


def test_validate_corpus(
    multilingual_ipa_tg_corpus_dir, english_mfa_acoustic_model, english_us_mfa_dictionary, temp_dir
):
    if sys.platform == "win32":
        pytest.skip("Transcription testing uses ngram")
    command = [
        "validate",
        multilingual_ipa_tg_corpus_dir,
        english_us_mfa_dictionary,
        english_mfa_acoustic_model,
        "-t",
        temp_dir,
        "-q",
        "--clean",
        "--disable_mp",
        "--test_transcriptions",
    ]
    args, unknown = parser.parse_known_args(command)
    run_validate_corpus(args)


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
        "--config_path",
        mono_train_config_path,
    ]
    args, unknown = parser.parse_known_args(command)
    run_validate_corpus(args)


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
    args, unknown = parser.parse_known_args(command)
    if sys.platform == "win32":
        with pytest.raises(ArgumentError):
            run_validate_dictionary(args)
    else:
        run_validate_dictionary(args)


def test_validate_dictionary_train(
    basic_dict_path,
    temp_dir,
):

    command = [
        "validate_dictionary",
        basic_dict_path,
        "-t",
        os.path.join(temp_dir, "dictionary_validation"),
    ]
    args, unknown = parser.parse_known_args(command)
    if sys.platform == "win32":
        with pytest.raises(ArgumentError):
            run_validate_dictionary(args)
    else:
        run_validate_dictionary(args)
