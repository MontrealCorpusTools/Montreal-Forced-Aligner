
import os
import pytest


@pytest.fixture(scope='session')
def test_dir():
    return os.path.abspath('tests/data')

@pytest.fixture(scope='session')
def generated_dir(test_dir):
    generated = os.path.join(test_dir, 'generated')
    if not os.path.exists(generated):
        os.makedirs(generated)
    return generated

@pytest.fixture(scope='session')
def basic_dir(test_dir):
    return os.path.join(test_dir, 'basic')

@pytest.fixture(scope='session')
def dict_dir(test_dir):
    return os.path.join(test_dir, 'dictionaries')

@pytest.fixture(scope='session')
def basic_dict_path(dict_dir):
    return os.path.join(dict_dir, 'basic.txt')

@pytest.fixture(scope='session')
def acoustic_corpus_wav_path(basic_dir):
    return os.path.join(basic_dir, 'acoustic_corpus.wav')

@pytest.fixture(scope='session')
def acoustic_corpus_lab_path(basic_dir):
    return os.path.join(basic_dir, 'acoustic_corpus.lab')

@pytest.fixture(scope='session')
def acoustic_corpus_textgrid_path(basic_dir):
    return os.path.join(basic_dir, 'acoustic_corpus.TextGrid')
