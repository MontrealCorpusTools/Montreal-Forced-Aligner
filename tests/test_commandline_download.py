import os
from aligner.command_line.download import run_download, get_pretrained_acoustic_path, \
    get_pretrained_g2p_path, get_dictionary_path, list_available_languages


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
    assert 'fr' in langs
    assert 'de' in langs


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



