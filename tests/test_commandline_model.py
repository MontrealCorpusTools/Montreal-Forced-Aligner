import os

import click.testing
import pytest

from montreal_forced_aligner.command_line.mfa import mfa_cli
from montreal_forced_aligner.exceptions import RemoteModelNotFoundError
from montreal_forced_aligner.models import AcousticModel, DictionaryModel, G2PModel, ModelManager


def test_get_available_languages():
    manager = ModelManager()
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


def test_download():
    command = [
        "model",
        "download",
        "acoustic",
        "sdsdsadad",
    ]
    with pytest.raises(RemoteModelNotFoundError):
        click.testing.CliRunner().invoke(mfa_cli, command, catch_exceptions=False)

    command = [
        "model",
        "download",
        "acoustic",
        "english_us_arpa",
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

    assert os.path.exists(AcousticModel.get_pretrained_path("english_us_arpa"))

    command = [
        "model",
        "download",
        "g2p",
        "english_us_arpa",
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

    assert os.path.exists(G2PModel.get_pretrained_path("english_us_arpa"))

    command = [
        "model",
        "download",
        "dictionary",
        "english_us_arpa",
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

    assert os.path.exists(DictionaryModel.get_pretrained_path("english_us_arpa"))

    command = ["model", "download", "acoustic", "--ignore_cache"]

    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
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
        "download",
        "dictionary",
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


def test_inspect_model():
    command = [
        "model",
        "inspect",
        "acoustic",
        "english_us_arpa",
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


def test_list_model():
    command = [
        "model",
        "list",
        "acoustic",
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


def test_save_model(transcription_acoustic_model):
    command = [
        "model",
        "save",
        "acoustic",
        transcription_acoustic_model,
        "--name",
        "test_acoustic",
        "--overwrite",
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
    assert os.path.exists(AcousticModel.get_pretrained_path("test_acoustic"))

    command = ["model", "inspect", "acoustic", "test_acoustic"]

    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
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

    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    assert isinstance(result.exception, SystemExit)

    command = ["model", "download", "acoustic", "not_bulgarian"]

    with pytest.raises(RemoteModelNotFoundError):
        click.testing.CliRunner().invoke(mfa_cli, command, catch_exceptions=False)
