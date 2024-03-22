import os

import click.testing
import pytest
import sqlalchemy.orm

from montreal_forced_aligner import config
from montreal_forced_aligner.command_line.mfa import mfa_cli
from montreal_forced_aligner.dictionary import MultispeakerDictionary
from montreal_forced_aligner.exceptions import PhoneMismatchError, RemoteModelNotFoundError
from montreal_forced_aligner.models import AcousticModel, DictionaryModel, G2PModel, ModelManager


def test_get_available_languages():
    manager = ModelManager(ignore_cache=True)
    manager.refresh_remote()
    model_type = "acoustic"
    acoustic_models = manager.remote_models[model_type]
    assert "archive" not in acoustic_models
    assert "english_us_arpa" in acoustic_models

    model_type = "g2p"
    langs = manager.remote_models[model_type]
    assert "bulgarian_mfa" in langs

    model_type = "dictionary"
    langs = manager.remote_models[model_type]
    assert "english_us_arpa" in langs
    assert "vietnamese_cv" in langs
    assert "vietnamese_mfa" in langs


def test_download_error():
    command = [
        "model",
        "download",
        "acoustic",
        "sdsdsadad",
    ]
    with pytest.raises(RemoteModelNotFoundError):
        click.testing.CliRunner().invoke(mfa_cli, command, catch_exceptions=False)


def test_download_acoustic():
    command = ["model", "download", "acoustic", "german_mfa", "--ignore_cache"]

    result = click.testing.CliRunner(mix_stderr=False).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value
    path = AcousticModel.get_pretrained_path("german_mfa")
    assert path.exists()

    assert AcousticModel(path).version == "3.0.0"

    command = ["model", "download", "acoustic", "german_mfa", "--version", "2.0.0"]

    result = click.testing.CliRunner(mix_stderr=False).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value
    path = AcousticModel.get_pretrained_path("german_mfa")
    assert path.exists()

    assert AcousticModel(path).version != "3.0.0"


def test_download_g2p():
    command = [
        "model",
        "download",
        "g2p",
        "german_mfa",
    ]
    result = click.testing.CliRunner(mix_stderr=False).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value

    assert os.path.exists(G2PModel.get_pretrained_path("german_mfa"))


def test_download_dictionary():
    command = [
        "model",
        "download",
        "dictionary",
        "german_mfa",
    ]

    result = click.testing.CliRunner(mix_stderr=False).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value

    assert os.path.exists(DictionaryModel.get_pretrained_path("german_mfa"))


def test_download_list_acoustic():
    command = ["model", "download", "acoustic", "--ignore_cache"]

    result = click.testing.CliRunner(mix_stderr=False).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value


def test_download_list_dictionary():
    command = [
        "model",
        "download",
        "dictionary",
    ]

    result = click.testing.CliRunner(mix_stderr=False).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value


def test_inspect_model():
    command = [
        "model",
        "inspect",
        "acoustic",
        "english_us_arpa",
    ]
    result = click.testing.CliRunner(mix_stderr=False).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value


def test_add_pronunciations(
    hindi_dict_path, japanese_dict_path, basic_dict_path, acoustic_dict_path, db_setup
):
    config.CLEAN = True
    command = [
        "model",
        "save",
        "dictionary",
        str(hindi_dict_path),
        "--name",
        "hindi",
        "--overwrite",
    ]
    result = click.testing.CliRunner(mix_stderr=False).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value
    assert os.path.exists(DictionaryModel.get_pretrained_path("hindi"))

    sqlalchemy.orm.close_all_sessions()
    with pytest.raises(PhoneMismatchError):
        command = [
            "model",
            "add_words",
            "hindi",
            str(japanese_dict_path),
        ]
        result = click.testing.CliRunner(mix_stderr=False).invoke(
            mfa_cli, command, catch_exceptions=True
        )
        print(result.stdout)
        print(result.stderr)
        if result.exception:
            print(result.exc_info)
            raise result.exception
        assert not result.return_value
    command = [
        "model",
        "save",
        "dictionary",
        str(acoustic_dict_path),
        "--name",
        "acoustic",
        "--overwrite",
    ]
    result = click.testing.CliRunner(mix_stderr=False).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value
    assert os.path.exists(DictionaryModel.get_pretrained_path("acoustic"))
    sqlalchemy.orm.close_all_sessions()

    command = [
        "model",
        "add_words",
        "acoustic",
        str(basic_dict_path),
    ]
    result = click.testing.CliRunner(mix_stderr=False).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value

    pretrained_acoustic_path = DictionaryModel.get_pretrained_path("acoustic")
    assert pretrained_acoustic_path.exists()
    d = MultispeakerDictionary(pretrained_acoustic_path)
    d.dictionary_setup()

    assert "hopefully" in d.word_mapping()
    d.cleanup_connections()


def test_list_model():
    command = [
        "model",
        "list",
        "acoustic",
    ]
    result = click.testing.CliRunner(mix_stderr=False).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value


def test_save_model(transcription_acoustic_model):
    command = [
        "model",
        "save",
        "acoustic",
        str(transcription_acoustic_model),
        "--name",
        "test_acoustic",
        "--overwrite",
    ]
    result = click.testing.CliRunner(mix_stderr=False).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value
    assert os.path.exists(AcousticModel.get_pretrained_path("test_acoustic"))

    command = ["model", "inspect", "acoustic", "test_acoustic"]

    result = click.testing.CliRunner(mix_stderr=False).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value


def test_expected_errors():
    command = ["model", "download", "not_acoustic", "bulgarian"]

    result = click.testing.CliRunner(mix_stderr=False).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    assert isinstance(result.exception, SystemExit)

    command = ["model", "download", "acoustic", "not_bulgarian"]

    with pytest.raises(RemoteModelNotFoundError):
        click.testing.CliRunner().invoke(mfa_cli, command, catch_exceptions=False)
