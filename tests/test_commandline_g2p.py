import os

import click.testing
import sqlalchemy.orm

from montreal_forced_aligner.command_line.mfa import mfa_cli
from montreal_forced_aligner.dictionary import MultispeakerDictionary


def test_generate_pretrained(
    english_g2p_model, basic_corpus_dir, temp_dir, generated_dir, db_setup
):
    output_path = generated_dir.joinpath("g2p_out.txt")
    command = [
        "g2p",
        basic_corpus_dir,
        english_g2p_model,
        output_path,
        "-q",
        "--clean",
        "--num_pronunciations",
        "1",
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
    assert os.path.exists(output_path)
    sqlalchemy.orm.close_all_sessions()
    d = MultispeakerDictionary(output_path)
    d.dictionary_setup()
    assert d.num_speech_words > 0


def test_generate_pretrained_dictionary(
    english_g2p_model, combined_corpus_dir, english_dictionary, temp_dir, generated_dir, db_setup
):
    output_path = generated_dir.joinpath("filtered_g2p_out.txt")
    command = [
        "g2p",
        combined_corpus_dir,
        english_g2p_model,
        output_path,
        "-q",
        "--clean",
        "--dictionary_path",
        english_dictionary,
        "--num_pronunciations",
        "1",
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
    assert os.path.exists(output_path)
    sqlalchemy.orm.close_all_sessions()
    d = MultispeakerDictionary(output_path)
    d.dictionary_setup()
    assert d.num_speech_words == 2


def test_generate_pretrained_threshold(
    english_g2p_model, basic_corpus_dir, temp_dir, generated_dir, db_setup
):
    output_path = generated_dir.joinpath("g2p_out.txt")
    command = [
        "g2p",
        basic_corpus_dir,
        english_g2p_model,
        output_path,
        "-q",
        "--clean",
        "--g2p_threshold",
        "0.95",
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
    sqlalchemy.orm.close_all_sessions()
    d = MultispeakerDictionary(output_path)
    d.dictionary_setup()

    assert d.num_speech_words > 0


def test_generate_pretrained_corpus(
    english_g2p_model, basic_corpus_dir, temp_dir, generated_dir, db_setup
):
    output_path = generated_dir.joinpath("g2p_directory_output")
    command = [
        "g2p",
        basic_corpus_dir,
        english_g2p_model,
        output_path,
        "-q",
        "--clean",
        "--num_pronunciations",
        "2",
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

    output_paths = [
        os.path.join(output_path, "michael", "acoustic corpus.lab"),
        os.path.join(output_path, "michael", "acoustic_corpus.lab"),
        os.path.join(output_path, "sickmichael", "cold corpus.lab"),
        os.path.join(output_path, "sickmichael", "cold_corpus.lab"),
        os.path.join(output_path, "sickmichael", "cold corpus3.lab"),
        os.path.join(output_path, "sickmichael", "cold_corpus3.lab"),
    ]

    for path in output_paths:
        assert os.path.exists(path)


def test_train_g2p(
    basic_dict_path,
    basic_g2p_model_path,
    temp_dir,
    train_g2p_config_path,
    db_setup,
):
    command = [
        "train_g2p",
        basic_dict_path,
        basic_g2p_model_path,
        "-q",
        "--clean",
        "--debug",
        "--validate",
        "--config_path",
        train_g2p_config_path,
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
    assert os.path.exists(basic_g2p_model_path)


def test_train_g2p_phonetisaurus(
    basic_dict_path,
    basic_phonetisaurus_g2p_model_path,
    temp_dir,
    train_g2p_config_path,
    db_setup,
):
    command = [
        "train_g2p",
        basic_dict_path,
        basic_phonetisaurus_g2p_model_path,
        "-q",
        "--clean",
        "--debug",
        "--validate",
        "--phonetisaurus" "--config_path",
        train_g2p_config_path,
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
        "-q",
        "--clean",
        "--debug",
        "--config_path",
        g2p_config_path,
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
    assert os.path.exists(g2p_basic_output)
    sqlalchemy.orm.close_all_sessions()
    d = MultispeakerDictionary(dictionary_path=g2p_basic_output)
    d.dictionary_setup()
    assert d.num_speech_words > 0


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
        "-q",
        "--clean",
        "--debug",
        "--config_path",
        g2p_config_path,
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
    assert os.path.exists(g2p_basic_phonetisaurus_output)
    sqlalchemy.orm.close_all_sessions()
    d = MultispeakerDictionary(dictionary_path=g2p_basic_phonetisaurus_output)
    d.dictionary_setup()
    assert d.num_speech_words > 0


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
        "-q",
        "--clean",
        "--debug",
        "--config_path",
        g2p_config_path,
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
    assert os.path.exists(g2p_basic_output)
    sqlalchemy.orm.close_all_sessions()
    d = MultispeakerDictionary(dictionary_path=g2p_basic_output)
    d.dictionary_setup()
    assert d.num_speech_words > 0


def test_generate_dict_textgrid(
    multilingual_ipa_tg_corpus_dir,
    english_g2p_model,
    generated_dir,
    temp_dir,
    g2p_config_path,
    db_setup,
):
    output_file = generated_dir.joinpath("tg_g2pped.dict")
    command = [
        "g2p",
        multilingual_ipa_tg_corpus_dir,
        english_g2p_model,
        output_file,
        "-q",
        "--clean",
        "--debug",
        "--config_path",
        g2p_config_path,
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
    assert os.path.exists(output_file)
    sqlalchemy.orm.close_all_sessions()
    d = MultispeakerDictionary(dictionary_path=output_file)
    d.dictionary_setup()
    assert d.num_speech_words > 0
