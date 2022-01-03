import os

import pytest

from montreal_forced_aligner.command_line.g2p import run_g2p
from montreal_forced_aligner.command_line.mfa import parser
from montreal_forced_aligner.command_line.train_g2p import run_train_g2p
from montreal_forced_aligner.dictionary.pronunciation import PronunciationDictionary
from montreal_forced_aligner.g2p.generator import G2P_DISABLED


def test_generate_pretrained(english_g2p_model, basic_corpus_dir, temp_dir, generated_dir):
    if G2P_DISABLED:
        pytest.skip("No Pynini found")
    output_path = os.path.join(generated_dir, "g2p_out.txt")
    command = [
        "g2p",
        "english_g2p",
        basic_corpus_dir,
        output_path,
        "-t",
        temp_dir,
        "-q",
        "--clean",
        "--num_pronunciations",
        "1",
        "--use_mp",
        "False",
    ]
    args, unknown = parser.parse_known_args(command)
    run_g2p(args, unknown)
    assert os.path.exists(output_path)
    d = PronunciationDictionary(output_path, temporary_directory=temp_dir)

    assert len(d.words) > 0


def test_generate_pretrained_threshold(
    english_g2p_model, basic_corpus_dir, temp_dir, generated_dir
):
    if G2P_DISABLED:
        pytest.skip("No Pynini found")
    output_path = os.path.join(generated_dir, "g2p_out.txt")
    command = [
        "g2p",
        "english_g2p",
        basic_corpus_dir,
        output_path,
        "-t",
        temp_dir,
        "-q",
        "--clean",
        "--g2p_threshold",
        "0.95",
    ]
    args, unknown = parser.parse_known_args(command)
    run_g2p(args, unknown)
    assert os.path.exists(output_path)
    d = PronunciationDictionary(output_path, temporary_directory=temp_dir)

    assert len(d.words) > 0


def test_train_g2p(sick_dict_path, sick_g2p_model_path, temp_dir, train_g2p_config_path):
    if G2P_DISABLED:
        pytest.skip("No Pynini found")
    command = [
        "train_g2p",
        sick_dict_path,
        sick_g2p_model_path,
        "-t",
        os.path.join(temp_dir, "test_train_g2p"),
        "-q",
        "--clean",
        "--debug",
        "--validate",
        "--config_path",
        train_g2p_config_path,
    ]
    args, unknown = parser.parse_known_args(command)
    run_train_g2p(args, unknown)
    assert os.path.exists(sick_g2p_model_path)


def test_generate_dict(
    basic_corpus_dir,
    sick_g2p_model_path,
    g2p_sick_output,
    temp_dir,
    g2p_config_path,
):
    if G2P_DISABLED:
        pytest.skip("No Pynini found")
    command = [
        "g2p",
        sick_g2p_model_path,
        basic_corpus_dir,
        g2p_sick_output,
        "-t",
        temp_dir,
        "-q",
        "--clean",
        "--debug",
        "--config_path",
        g2p_config_path,
    ]
    args, unknown = parser.parse_known_args(command)
    run_g2p(args, unknown)
    assert os.path.exists(g2p_sick_output)
    d = PronunciationDictionary(dictionary_path=g2p_sick_output, temporary_directory=temp_dir)
    assert len(d.words) > 0


def test_generate_dict_text_only(
    basic_split_dir,
    sick_g2p_model_path,
    g2p_sick_output,
    temp_dir,
    g2p_config_path,
):
    if G2P_DISABLED:
        pytest.skip("No Pynini found")
    text_dir = basic_split_dir[1]
    command = [
        "g2p",
        sick_g2p_model_path,
        text_dir,
        g2p_sick_output,
        "-t",
        temp_dir,
        "-q",
        "--clean",
        "--debug",
        "--config_path",
        g2p_config_path,
    ]
    args, unknown = parser.parse_known_args(command)
    run_g2p(args, unknown)
    assert os.path.exists(g2p_sick_output)
    d = PronunciationDictionary(dictionary_path=g2p_sick_output, temporary_directory=temp_dir)
    assert len(d.words) > 0


def test_generate_orthography_dict(basic_corpus_dir, orth_sick_output, temp_dir):
    if G2P_DISABLED:
        pytest.skip("No Pynini found")
    command = [
        "g2p",
        basic_corpus_dir,
        orth_sick_output,
        "-t",
        temp_dir,
        "-q",
        "--clean",
        "--debug",
        "--use_mp",
        "False",
    ]
    args, unknown = parser.parse_known_args(command)
    run_g2p(args, unknown)
    assert os.path.exists(orth_sick_output)
    d = PronunciationDictionary(dictionary_path=orth_sick_output, temporary_directory=temp_dir)
    assert len(d.words) > 0
