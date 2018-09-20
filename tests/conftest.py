from aligner.command_line.align import fix_path

fix_path()

import os
import shutil
import pytest

from aligner.corpus import Corpus
from aligner.dictionary import Dictionary
from aligner.config import train_yaml_to_config


def pytest_addoption(parser):
    parser.addoption("--skiplarge", action="store_true",
                     help="skip large dataset tests")


@pytest.fixture(scope='session')
def test_dir():
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, 'data')


@pytest.fixture(scope='session')
def wav_dir(test_dir):
    return os.path.join(test_dir, 'wav')


@pytest.fixture(scope='session')
def lab_dir(test_dir):
    return os.path.join(test_dir, 'lab')


@pytest.fixture(scope='session')
def textgrid_dir(test_dir):
    return os.path.join(test_dir, 'textgrid')


@pytest.fixture(scope='session')
def generated_dir(test_dir):
    generated = os.path.join(test_dir, 'generated')
    shutil.rmtree(generated, ignore_errors=True)
    if not os.path.exists(generated):
        os.makedirs(generated)
    return generated


@pytest.fixture(scope='session')
def temp_dir(generated_dir):
    return os.path.join(generated_dir, 'temp')


@pytest.fixture(scope='session')
def corpus_root_dir(generated_dir):
    return os.path.join(generated_dir, 'corpus')


@pytest.fixture(scope='session')
def basic_corpus_dir(corpus_root_dir, wav_dir, lab_dir):
    path = os.path.join(corpus_root_dir, 'basic')
    os.makedirs(path, exist_ok=True)
    names = [('michael', ['acoustic_corpus']), ('sickmichael', ['cold_corpus', 'cold_corpus3'])]
    for s, files in names:
        s_dir = os.path.join(path, s)
        os.makedirs(s_dir, exist_ok=True)
        for name in files:
            shutil.copyfile(os.path.join(wav_dir, name + '.wav'), os.path.join(s_dir, name + '.wav'))
            shutil.copyfile(os.path.join(lab_dir, name + '.lab'), os.path.join(s_dir, name + '.lab'))
    return path


@pytest.fixture(scope='session')
def basic_corpus_txt_dir(corpus_root_dir, wav_dir, lab_dir):
    path = os.path.join(corpus_root_dir, 'basic_txt')
    os.makedirs(path, exist_ok=True)
    names = [('michael', ['acoustic_corpus']), ('sickmichael', ['cold_corpus', 'cold_corpus3'])]
    for s, files in names:
        s_dir = os.path.join(path, s)
        os.makedirs(s_dir, exist_ok=True)
        for name in files:
            shutil.copyfile(os.path.join(wav_dir, name + '.wav'), os.path.join(s_dir, name + '.wav'))
            shutil.copyfile(os.path.join(lab_dir, name + '.lab'), os.path.join(s_dir, name + '.txt'))
    return path


@pytest.fixture(scope='session')
def extra_corpus_dir(corpus_root_dir, wav_dir, lab_dir):
    path = os.path.join(corpus_root_dir, 'extra')
    os.makedirs(path, exist_ok=True)
    name = 'cold_corpus3'
    shutil.copyfile(os.path.join(wav_dir, name + '.wav'), os.path.join(path, name + '.wav'))
    shutil.copyfile(os.path.join(lab_dir, name + '_extra.lab'), os.path.join(path, name + '.lab'))
    return path


@pytest.fixture(scope='session')
def stereo_corpus_dir(corpus_root_dir, wav_dir, textgrid_dir):
    path = os.path.join(corpus_root_dir, 'stereo')
    os.makedirs(path, exist_ok=True)
    name = 'michaelandsickmichael'
    shutil.copyfile(os.path.join(wav_dir, name + '.wav'), os.path.join(path, name + '.wav'))
    shutil.copyfile(os.path.join(textgrid_dir, name + '.TextGrid'), os.path.join(path, name + '.TextGrid'))
    return path


@pytest.fixture(scope='session')
def shortsegments_corpus_dir(corpus_root_dir, wav_dir, textgrid_dir):
    path = os.path.join(corpus_root_dir, 'short_segments')
    os.makedirs(path, exist_ok=True)
    name = 'short_segments'
    shutil.copyfile(os.path.join(wav_dir, 'dummy.wav'), os.path.join(path, name + '.wav'))
    shutil.copyfile(os.path.join(textgrid_dir, name + '.TextGrid'), os.path.join(path, name + '.TextGrid'))
    return path


@pytest.fixture(scope='session')
def vietnamese_corpus_dir(corpus_root_dir, wav_dir, textgrid_dir):
    path = os.path.join(corpus_root_dir, 'vietnamese')
    os.makedirs(path, exist_ok=True)
    name = 'vietnamese'
    shutil.copyfile(os.path.join(wav_dir, 'dummy.wav'), os.path.join(path, name + '.wav'))
    shutil.copyfile(os.path.join(textgrid_dir, name + '.TextGrid'), os.path.join(path, name + '.TextGrid'))
    return path


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


# @pytest.fixture(scope='session')
# def basic_roots_path(expected_dict_path):
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
def sick_corpus(basic_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, 'sickcorpus')
    corpus = Corpus(basic_corpus_dir, output_directory, num_jobs=2)
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
def prosodylab_output_directory(generated_dir):
    return os.path.join(generated_dir, 'prosodylab_output')


@pytest.fixture(scope='session')
def textgrid_output_directory(generated_dir):
    return os.path.join(generated_dir, 'textgrid_output')


@pytest.fixture(scope='session')
def single_speaker_prosodylab_format_directory(large_prosodylab_format_directory):
    return os.path.join(large_prosodylab_format_directory, '121')


@pytest.fixture(scope='session')
def single_speaker_textgrid_format_directory(large_textgrid_format_directory):
    return os.path.join(large_textgrid_format_directory, '121')


@pytest.fixture(scope='session')
def prosodylab_output_model_path(generated_dir):
    return os.path.join(generated_dir, 'prosodylab_output_model.zip')


@pytest.fixture(scope='session')
def textgrid_output_model_path(generated_dir):
    return os.path.join(generated_dir, 'textgrid_output_model.zip')


@pytest.fixture(scope='session')
def training_dict_path(test_dir):
    return os.path.join(test_dir, "dictionaries", "chinese_dict.txt", )


@pytest.fixture(scope='session')
def g2p_model_path(generated_dir):
    return os.path.join(generated_dir, 'pinyin_g2p.zip')


@pytest.fixture(scope='session')
def sick_g2p_model_path(generated_dir):
    return os.path.join(generated_dir, 'sick_g2p.zip')


@pytest.fixture(scope='session')
def g2p_sick_output(generated_dir):
    return os.path.join(generated_dir, 'g2p_sick.txt')


@pytest.fixture(scope='session')
def orth_sick_output(generated_dir):
    return os.path.join(generated_dir, 'orth_sick.txt')


@pytest.fixture(scope='session')
def example_output_model_path(generated_dir):
    return os.path.join(generated_dir, 'example_output_model.zip')


@pytest.fixture(scope='session')
def KO_dict(test_dir):
    return os.path.join(test_dir, "dictionaries", "KO_dict.txt")


@pytest.fixture(scope='session')
def config_directory(test_dir):
    return os.path.join(test_dir, 'configs')


@pytest.fixture(scope='session')
def basic_train_config(config_directory):
    return os.path.join(config_directory, 'basic_train_config.yaml')


@pytest.fixture(scope='session')
def basic_align_config(config_directory):
    return os.path.join(config_directory, 'basic_align_config.yaml')


@pytest.fixture(scope='session')
def mono_train_config(config_directory):
    return train_yaml_to_config(os.path.join(config_directory, 'mono_train.yaml'))


@pytest.fixture(scope='session')
def tri_train_config(config_directory):
    return train_yaml_to_config(os.path.join(config_directory, 'tri_train.yaml'))


@pytest.fixture(scope='session')
def lda_train_config(config_directory):
    return train_yaml_to_config(os.path.join(config_directory, 'lda_train.yaml'))


@pytest.fixture(scope='session')
def sat_train_config(config_directory):
    return train_yaml_to_config(os.path.join(config_directory, 'sat_train.yaml'))


@pytest.fixture(scope='session')
def lda_sat_train_config(config_directory):
    return train_yaml_to_config(os.path.join(config_directory, 'lda_sat_train.yaml'))


@pytest.fixture(scope='session')
def ivector_train_config(config_directory):
    return train_yaml_to_config(os.path.join(config_directory, 'ivector_train.yaml'))


@pytest.fixture(scope='session')
def nnet_train_config(config_directory):
    return train_yaml_to_config(os.path.join(config_directory, 'nnet_train.yaml'))


@pytest.fixture(scope='session')
def nnet_ivectors_train_config(config_directory):
    return train_yaml_to_config(os.path.join(config_directory, 'nnet_ivectors_train.yaml'))
