
from aligner.command_line.align import fix_path

fix_path()

import os
import pytest

from aligner.corpus import Corpus
from aligner.dictionary import Dictionary


def pytest_addoption(parser):
    parser.addoption("--skiplarge", action="store_true",
        help="skip large dataset tests")


@pytest.fixture(scope='session')
def test_dir():
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, 'data')


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
def extra_dir(test_dir):
    return os.path.join(test_dir, 'extra')

@pytest.fixture(scope='session')
def dict_dir(test_dir):
    return os.path.join(test_dir, 'dictionaries')


@pytest.fixture(scope='session')
def basic_dict_path(dict_dir):
    return os.path.join(dict_dir, 'basic.txt')


@pytest.fixture(scope='session')
def extra_annotations_path(dict_dir):
    return os.path.join(dict_dir, 'extra_annotations.txt')


@pytest.fixture(scope='session')
def frclitics_dict_path(dict_dir):
    return os.path.join(dict_dir, 'frclitics.txt')


@pytest.fixture(scope='session')
def expected_dict_path(dict_dir):
    return os.path.join(dict_dir, 'expected')


@pytest.fixture(scope='session')
def basic_topo_path(expected_dict_path):
    return os.path.join(expected_dict_path, 'topo')


@pytest.fixture(scope='session')
def basic_graphemes_path(expected_dict_path):
    return os.path.join(expected_dict_path, 'graphemes.txt')


@pytest.fixture(scope='session')
def basic_phone_map_path(expected_dict_path):
    return os.path.join(expected_dict_path, 'phone_map.txt')


@pytest.fixture(scope='session')
def basic_phones_path(expected_dict_path):
    return os.path.join(expected_dict_path, 'phones.txt')


@pytest.fixture(scope='session')
def basic_words_path(expected_dict_path):
    return os.path.join(expected_dict_path, 'words.txt')


@pytest.fixture(scope='session')
def basic_rootsint_path(expected_dict_path):
    return os.path.join(expected_dict_path, 'roots.int')


@pytest.fixture(scope='session')
def basic_rootstxt_path(expected_dict_path):
    return os.path.join(expected_dict_path, 'roots.txt')


#@pytest.fixture(scope='session')
#def basic_roots_path(expected_dict_path):
#    return os.path.join(expected_dict_path, 'roots.txt')


@pytest.fixture(scope='session')
def basic_setsint_path(expected_dict_path):
    return os.path.join(expected_dict_path, 'sets.int')


@pytest.fixture(scope='session')
def basic_setstxt_path(expected_dict_path):
    return os.path.join(expected_dict_path, 'sets.txt')


@pytest.fixture(scope='session')
def basic_word_boundaryint_path(expected_dict_path):
    return os.path.join(expected_dict_path, 'word_boundary.int')


@pytest.fixture(scope='session')
def basic_word_boundarytxt_path(expected_dict_path):
    return os.path.join(expected_dict_path, 'word_boundary.txt')


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
def michael_corpus_lab_path(basic_dir):
    return os.path.join(basic_dir, 'michael_corpus.lab')


@pytest.fixture(scope='session')
def output_directory(basic_dir):
    return os.path.join(basic_dir, 'output')


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
    corpus = Corpus(basic_dir, output_directory, num_jobs = 2)
    corpus.write()
    corpus.create_mfccs()
    corpus.setup_splits(sick_dict)
    return corpus


@pytest.fixture(scope='session')
def textgrid_directory(test_dir):
    return os.path.join(test_dir, 'textgrid')


@pytest.fixture(scope='session')
def large_dataset_directory():
    if os.environ.get('TRAVIS', False):
        directory = os.path.expanduser('~/tools/mfa_test_data')
    else:
        test_dir = os.path.dirname(os.path.abspath(__file__))
        repo_dir = os.path.dirname(test_dir)
        root_dir = os.path.dirname(repo_dir)
        directory = os.path.join(root_dir, 'mfa_test_data')
    if not os.path.exists(directory):
        pytest.skip('Couldn\'t find the mfa_test_data directory')
    else:
        return directory


@pytest.fixture(scope='session')
def large_dataset_dictionary(large_dataset_directory):
    return os.path.join(large_dataset_directory, 'librispeech-lexicon.txt')


@pytest.fixture(scope='session')
def large_prosodylab_format_directory(large_dataset_directory):
    return os.path.join(large_dataset_directory, 'prosodylab_format')


@pytest.fixture(scope='session')
def large_textgrid_format_directory(large_dataset_directory):
    return os.path.join(large_dataset_directory, 'textgrid_format')


@pytest.fixture(scope='session')
def prosodylab_output_directory():
    return os.path.expanduser('~/prosodylab_output')


@pytest.fixture(scope='session')
def textgrid_output_directory():
    return os.path.expanduser('~/textgrid_output')


@pytest.fixture(scope='session')
def single_speaker_prosodylab_format_directory(large_prosodylab_format_directory):
    return os.path.join(large_prosodylab_format_directory, '121')


@pytest.fixture(scope='session')
def single_speaker_textgrid_format_directory(large_textgrid_format_directory):
    return os.path.join(large_textgrid_format_directory, '121')


@pytest.fixture(scope='session')
def prosodylab_output_model_path():
    return os.path.expanduser('~/prosodylab_output_model.zip')


@pytest.fixture(scope='session')
def textgrid_output_model_path():
    return os.path.expanduser('~/textgrid_output_model.zip')
