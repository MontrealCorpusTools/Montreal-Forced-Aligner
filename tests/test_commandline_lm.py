import os
import sys

import pytest

from montreal_forced_aligner.command_line.mfa import parser
from montreal_forced_aligner.command_line.train_lm import run_train_lm


def test_train_lm(basic_corpus_dir, temp_dir, generated_dir, basic_train_lm_config_path):
    if sys.platform == "win32":
        pytest.skip("LM training not supported on Windows.")
    temp_dir = os.path.join(temp_dir, "train_lm")
    command = [
        "train_lm",
        basic_corpus_dir,
        os.path.join(generated_dir, "test_basic_lm.zip"),
        "-t",
        temp_dir,
        "--config_path",
        basic_train_lm_config_path,
        "-q",
        "--clean",
    ]
    args, unknown = parser.parse_known_args(command)
    run_train_lm(args)
    assert os.path.exists(args.output_model_path)


def test_train_lm_text(basic_split_dir, temp_dir, generated_dir, basic_train_lm_config_path):
    if sys.platform == "win32":
        pytest.skip("LM training not supported on Windows.")
    temp_dir = os.path.join(temp_dir, "train_lm_text")
    text_dir = basic_split_dir[1]
    command = [
        "train_lm",
        text_dir,
        os.path.join(generated_dir, "test_basic_lm_split.zip"),
        "-t",
        temp_dir,
        "--config_path",
        basic_train_lm_config_path,
        "-q",
        "--clean",
    ]
    args, unknown = parser.parse_known_args(command)
    run_train_lm(args)
    assert os.path.exists(args.output_model_path)


def test_train_lm_dictionary(
    basic_split_dir, sick_dict_path, temp_dir, generated_dir, basic_train_lm_config_path
):
    if sys.platform == "win32":
        pytest.skip("LM training not supported on Windows.")
    temp_dir = os.path.join(temp_dir, "train_lm_dictionary")
    text_dir = basic_split_dir[1]
    command = [
        "train_lm",
        text_dir,
        os.path.join(generated_dir, "test_basic_lm_split.zip"),
        "-t",
        temp_dir,
        "--dictionary_path",
        sick_dict_path,
        "--config_path",
        basic_train_lm_config_path,
        "-q",
        "--clean",
    ]
    args, unknown = parser.parse_known_args(command)
    run_train_lm(args)
    assert os.path.exists(args.output_model_path)


def test_train_lm_arpa(
    transcription_language_model_arpa, temp_dir, generated_dir, basic_train_lm_config_path
):
    if sys.platform == "win32":
        pytest.skip("LM training not supported on Windows.")
    temp_dir = os.path.join(temp_dir, "train_lm_arpa")
    command = [
        "train_lm",
        transcription_language_model_arpa,
        os.path.join(generated_dir, "test_basic_lm_split.zip"),
        "-t",
        temp_dir,
        "--config_path",
        basic_train_lm_config_path,
        "-q",
        "--clean",
    ]
    args, unknown = parser.parse_known_args(command)
    run_train_lm(args)
    assert os.path.exists(args.output_model_path)


def test_train_lm_text_no_mp(basic_split_dir, temp_dir, generated_dir, basic_train_lm_config_path):
    if sys.platform == "win32":
        pytest.skip("LM training not supported on Windows.")
    text_dir = basic_split_dir[1]
    command = [
        "train_lm",
        text_dir,
        os.path.join(generated_dir, "test_basic_lm_split.zip"),
        "-t",
        temp_dir,
        "--config_path",
        basic_train_lm_config_path,
        "-q",
        "--clean",
        "-j",
        "1",
    ]
    args, unknown = parser.parse_known_args(command)
    run_train_lm(args)
    assert os.path.exists(args.output_model_path)
