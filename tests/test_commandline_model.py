import os
import pytest
from argparse import Namespace
from montreal_forced_aligner.command_line.model import run_model, get_pretrained_path, \
    list_downloadable_languages, PretrainedModelNotFoundError, ModelTypeNotSupportedError


class DummyArgs(Namespace):
    def __init__(self):
        self.action = ''
        self.model_type = ''
        self.name = ''


def test_get_available_languages():
    model_type = 'acoustic'
    langs = list_downloadable_languages(model_type)
    assert 'bulgarian' in langs
    assert 'english' in langs

    model_type = 'g2p'
    langs = list_downloadable_languages(model_type)
    assert 'bulgarian_g2p' in langs

    model_type = 'dictionary'
    langs = list_downloadable_languages(model_type)
    assert 'english' in langs
    assert 'french_prosodylab' in langs
    assert 'german_prosodylab' in langs


def test_download():
    args = DummyArgs()
    args.action = 'download'
    args.name = 'bulgarian'
    args.model_type = 'acoustic'

    run_model(args)

    assert os.path.exists(get_pretrained_path('acoustic', args.name))

    args = DummyArgs()
    args.action = 'download'
    args.name = 'bulgarian_g2p'
    args.model_type = 'g2p'

    run_model(args)

    assert os.path.exists(get_pretrained_path('g2p', args.name))

    args = DummyArgs()
    args.action = 'download'
    args.name = 'english'
    args.model_type = 'dictionary'

    run_model(args)

    assert os.path.exists(get_pretrained_path('dictionary', args.name))

    args = DummyArgs()
    args.action = 'download'
    args.name = ''
    args.model_type = 'dictionary'

    run_model(args)


def test_expected_errors():
    args = DummyArgs()
    args.action = 'download'
    args.name = 'bulgarian'
    args.model_type = 'not_acoustic'
    with pytest.raises(ModelTypeNotSupportedError):
        run_model(args)

    args = DummyArgs()
    args.action = 'download'
    args.name = 'not_bulgarian'
    args.model_type = 'acoustic'
    with pytest.raises(PretrainedModelNotFoundError):
        run_model(args)



