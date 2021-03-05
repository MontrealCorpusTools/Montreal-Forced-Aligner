import os
import pytest
from montreal_forced_aligner.command_line.download import run_download, get_pretrained_acoustic_path, \
    get_pretrained_g2p_path, get_dictionary_path, list_available_languages, ArgumentError


class DummyArgs(object):
    def __init__(self):
        self.model_type = ''
        self.language = ''


def test_get_available_languages():
    model_type = 'acoustic'
    langs = list_available_languages(model_type)
    assert 'bulgarian' in langs
    assert 'english' in langs

    model_type = 'g2p'
    langs = list_available_languages(model_type)
    assert 'bulgarian_g2p' in langs

    model_type = 'dictionary'
    langs = list_available_languages(model_type)
    assert 'english' in langs
    assert 'french_prosodylab' in langs
    assert 'german_prosodylab' in langs


def test_download():
    args = DummyArgs()
    args.language = 'bulgarian'
    args.model_type = 'acoustic'

    run_download(args)

    assert os.path.exists(get_pretrained_acoustic_path(args.language))

    args = DummyArgs()
    args.language = 'bulgarian_g2p'
    args.model_type = 'g2p'

    run_download(args)

    assert os.path.exists(get_pretrained_g2p_path(args.language))

    args = DummyArgs()
    args.language = 'english'
    args.model_type = 'dictionary'

    run_download(args)

    assert os.path.exists(get_dictionary_path(args.language))

    args = DummyArgs()
    args.language = ''
    args.model_type = 'dictionary'

    run_download(args)


def test_expected_errors():
    args = DummyArgs()
    args.language = 'bulgarian'
    args.model_type = 'not_acoustic'
    with pytest.raises(ArgumentError):
        run_download(args)

    args = DummyArgs()
    args.language = 'not_bulgarian'
    args.model_type = 'acoustic'
    with pytest.raises(ArgumentError):
        run_download(args)



