import os

import click.testing

from montreal_forced_aligner.command_line.mfa import mfa_cli
from montreal_forced_aligner.command_line.utils import check_databases
from montreal_forced_aligner.dictionary import MultispeakerDictionary


def test_generate_pretrained(
    english_g2p_model, basic_corpus_dir, temp_dir, generated_dir, db_setup
):
    output_path = os.path.join(generated_dir, "g2p_out.txt")
    command = [
        "g2p",
        basic_corpus_dir,
        english_g2p_model,
        output_path,
        "-t",
        os.path.join(temp_dir, "g2p_cli"),
        "-q",
        "--clean",
        "--num_pronunciations",
        "1",
        "--use_mp",
        "False",
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
    assert os.path.exists(output_path)
    check_databases()
    d = MultispeakerDictionary(output_path)
    d.dictionary_setup()
    assert len(d.word_mapping(list(d.dictionary_lookup.values())[0])) > 0


def test_generate_pretrained_threshold(
    english_g2p_model, basic_corpus_dir, temp_dir, generated_dir, db_setup
):
    output_path = os.path.join(generated_dir, "g2p_out.txt")
    command = [
        "g2p",
        basic_corpus_dir,
        english_g2p_model,
        output_path,
        "-t",
        os.path.join(temp_dir, "g2p_cli"),
        "-q",
        "--clean",
        "--g2p_threshold",
        "0.95",
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
    assert os.path.exists(output_path)
    check_databases()
    d = MultispeakerDictionary(output_path)
    d.dictionary_setup()

    assert len(d.word_mapping(list(d.dictionary_lookup.values())[0])) > 0


def test_train_g2p(
    basic_dict_path,
    basic_g2p_model_path,
    temp_dir,
    train_g2p_config_path,
):
    command = [
        "train_g2p",
        basic_dict_path,
        basic_g2p_model_path,
        "-t",
        os.path.join(temp_dir, "test_train_g2p"),
        "-q",
        "--clean",
        "--debug",
        "--validate",
        "--config_path",
        train_g2p_config_path,
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
    assert os.path.exists(basic_g2p_model_path)


def test_train_g2p_phonetisaurus(
    basic_dict_path,
    basic_phonetisaurus_g2p_model_path,
    temp_dir,
    train_g2p_config_path,
):
    command = [
        "train_g2p",
        basic_dict_path,
        basic_phonetisaurus_g2p_model_path,
        "-t",
        os.path.join(temp_dir, "test_train_g2p"),
        "-q",
        "--clean",
        "--debug",
        "--validate",
        "--phonetisaurus" "--config_path",
        train_g2p_config_path,
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
    assert os.path.exists(basic_phonetisaurus_g2p_model_path)


def test_generate_dict(
    basic_corpus_dir,
    basic_g2p_model_path,
    g2p_basic_output,
    temp_dir,
    g2p_config_path,
    db_setup,
):
    command = [
        "g2p",
        basic_corpus_dir,
        basic_g2p_model_path,
        g2p_basic_output,
        "-t",
        os.path.join(temp_dir, "g2p_cli"),
        "-q",
        "--clean",
        "--debug",
        "--config_path",
        g2p_config_path,
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
    assert os.path.exists(g2p_basic_output)
    check_databases()
    d = MultispeakerDictionary(dictionary_path=g2p_basic_output)
    d.dictionary_setup()
    assert len(d.word_mapping(list(d.dictionary_lookup.values())[0])) > 0


def test_generate_dict_phonetisaurus(
    basic_corpus_dir,
    basic_phonetisaurus_g2p_model_path,
    g2p_basic_phonetisaurus_output,
    temp_dir,
    g2p_config_path,
    db_setup,
):
    command = [
        "g2p",
        basic_corpus_dir,
        basic_phonetisaurus_g2p_model_path,
        g2p_basic_phonetisaurus_output,
        "-t",
        os.path.join(temp_dir, "g2p_cli"),
        "-q",
        "--clean",
        "--debug",
        "--config_path",
        g2p_config_path,
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
    assert os.path.exists(g2p_basic_phonetisaurus_output)
    check_databases()
    d = MultispeakerDictionary(dictionary_path=g2p_basic_phonetisaurus_output)
    d.dictionary_setup()
    assert len(d.word_mapping(list(d.dictionary_lookup.values())[0])) > 0


def test_generate_dict_text_only(
    basic_split_dir,
    basic_g2p_model_path,
    g2p_basic_output,
    temp_dir,
    g2p_config_path,
    db_setup,
):
    text_dir = basic_split_dir[1]
    command = [
        "g2p",
        text_dir,
        basic_g2p_model_path,
        g2p_basic_output,
        "-t",
        os.path.join(temp_dir, "g2p_cli"),
        "-q",
        "--clean",
        "--debug",
        "--config_path",
        g2p_config_path,
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
    assert os.path.exists(g2p_basic_output)
    check_databases()
    d = MultispeakerDictionary(dictionary_path=g2p_basic_output)
    d.dictionary_setup()
    assert len(d.word_mapping(list(d.dictionary_lookup.values())[0])) > 0


def test_generate_dict_textgrid(
    multilingual_ipa_tg_corpus_dir,
    english_g2p_model,
    generated_dir,
    temp_dir,
    g2p_config_path,
    db_setup,
):
    output_file = os.path.join(generated_dir, "tg_g2pped.dict")
    command = [
        "g2p",
        multilingual_ipa_tg_corpus_dir,
        english_g2p_model,
        output_file,
        "-t",
        os.path.join(temp_dir, "g2p_cli"),
        "-q",
        "--clean",
        "--debug",
        "--config_path",
        g2p_config_path,
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
    assert os.path.exists(output_file)
    check_databases()
    d = MultispeakerDictionary(dictionary_path=output_file)
    d.dictionary_setup()
    assert len(d.word_mapping(list(d.dictionary_lookup.values())[0])) > 0
