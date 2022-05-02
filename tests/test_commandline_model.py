import os
from argparse import Namespace

import pytest

from montreal_forced_aligner.command_line.model import (
    ModelTypeNotSupportedError,
    RemoteModelNotFoundError,
    run_model,
)
from montreal_forced_aligner.models import AcousticModel, DictionaryModel, G2PModel, ModelManager


class DummyArgs(Namespace):
    def __init__(self):
        self.action = ""
        self.model_type = ""
        self.name = ""
        self.github_token = ""
        self.ignore_cache = False


def test_get_available_languages():
    manager = ModelManager()
    manager.refresh_remote()
    model_type = "acoustic"
    acoustic_models = manager.remote_models[model_type]
    print(manager.remote_models)
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
    args = DummyArgs()
    args.action = "download"
    args.name = "sdsdsadad"
    args.model_type = "acoustic"
    with pytest.raises(RemoteModelNotFoundError):
        run_model(args)

    args = DummyArgs()
    args.action = "download"
    args.name = "english_us_arpa"
    args.model_type = "acoustic"

    run_model(args)

    assert os.path.exists(AcousticModel.get_pretrained_path(args.name))

    args = DummyArgs()
    args.action = "download"
    args.name = "english_us_arpa"
    args.model_type = "g2p"

    run_model(args)

    assert os.path.exists(G2PModel.get_pretrained_path(args.name))

    args = DummyArgs()
    args.action = "download"
    args.name = "english_us_arpa"
    args.model_type = "dictionary"

    run_model(args)

    assert os.path.exists(DictionaryModel.get_pretrained_path(args.name))

    args = DummyArgs()
    args.action = "download"
    args.name = ""
    args.ignore_cache = True
    args.model_type = "dictionary"

    run_model(args)

    args = DummyArgs()
    args.action = "download"
    args.name = ""
    args.model_type = "dictionary"

    run_model(args)


def test_inspect_model():
    args = DummyArgs()
    args.action = "inspect"
    args.name = "english_us_arpa"
    args.model_type = "acoustic"
    run_model(args)


def test_list_model():
    args = DummyArgs()
    args.action = "list"
    args.model_type = "acoustic"
    run_model(args)


def test_save_model(transcription_acoustic_model):
    args = DummyArgs()
    args.action = "save"
    args.model_type = "acoustic"
    args.path = transcription_acoustic_model
    run_model(args)

    args = DummyArgs()
    args.action = "inspect"
    args.name = "mono_model"
    args.model_type = "acoustic"
    run_model(args)


def test_expected_errors():
    args = DummyArgs()
    args.action = "download"
    args.name = "bulgarian"
    args.model_type = "not_acoustic"
    with pytest.raises(ModelTypeNotSupportedError):
        run_model(args)

    args = DummyArgs()
    args.action = "download"
    args.name = "not_bulgarian"
    args.model_type = "acoustic"
    with pytest.raises(RemoteModelNotFoundError):
        run_model(args)
