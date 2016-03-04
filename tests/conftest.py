
import os
import pytest

from aligner.corpus import Corpus
from aligner.config import MfccConfig
from aligner.dictionary import Dictionary

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
def sick_dict_path(dict_dir):
    return os.path.join(dict_dir, 'sick.txt')

@pytest.fixture(scope='session')
def acoustic_corpus_wav_path(basic_dir):
    return os.path.join(basic_dir, 'acoustic_corpus.wav')

@pytest.fixture(scope='session')
def acoustic_corpus_lab_path(basic_dir):
    return os.path.join(basic_dir, 'acoustic_corpus.lab')

@pytest.fixture(scope='session')
def acoustic_corpus_textgrid_path(basic_dir):
    return os.path.join(basic_dir, 'acoustic_corpus.TextGrid')

@pytest.fixture(scope='session')
def sick_dict(sick_dict_path, generated_dir):
    output_directory = os.path.join(generated_dir, 'sickcorpus')
    dictionary = Dictionary(sick_dict_path, output_directory)
    dictionary.write()
    return dictionary

@pytest.fixture(scope='session')
def sick_corpus(sick_dict, basic_dir, generated_dir):
    output_directory = os.path.join(generated_dir, 'sickcorpus')
    c = MfccConfig(output_directory)
    corpus = Corpus(basic_dir, output_directory, c, num_jobs = 2)
    corpus.write()
    corpus.create_mfccs()
    corpus.setup_splits(sick_dict)
    return corpus


