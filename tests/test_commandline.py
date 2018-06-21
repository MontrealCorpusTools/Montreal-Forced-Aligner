import os
import pytest

from aligner.command_line.align import align_corpus, align_included_model

from aligner.command_line.train_and_align import align_corpus as train_and_align_corpus, align_corpus_no_dict
from aligner.command_line.generate_dictionary import generate_dict
from aligner.command_line.train_g2p import train_g2p

from aligner.exceptions import PronunciationAcousticMismatchError


class DummyArgs(object):
    def __init__(self):
        self.speaker_characters = 0
        self.num_jobs = 0
        self.verbose = False
        self.clean = True
        self.debug = False
        self.errors = False
        self.temp_directory = None
        self.quiet = True
        self.config_path = ''


class G2PDummyArgs(object):
    def __init__(self):
        self.temp_directory = None
        self.window_size = 2



large = pytest.mark.skipif(
    pytest.config.getoption("--skiplarge"),
    reason="remove --skiplarge option to run"
)


def assert_export_exist(old_directory, new_directory):
    for root, dirs, files in os.walk(old_directory):
        new_root = root.replace(old_directory, new_directory)
        for d in dirs:
            assert (os.path.exists(os.path.join(new_root, d)))
        for f in files:
            if not f.endswith('.wav'):
                continue
            new_f = f.replace('.wav', '.TextGrid')
            assert (os.path.exists(os.path.join(new_root, new_f)))


def test_align_basic(basic_corpus_dir, sick_dict_path, generated_dir, large_dataset_dictionary, temp_dir, basic_align_config):
    args = DummyArgs()
    args.acoustic_model_path = 'english'
    args.corpus_directory = basic_corpus_dir
    args.dictionary_path = sick_dict_path
    args.output_directory = os.path.join(generated_dir, 'basic_output')
    args.quiet = True
    args.temp_directory = temp_dir
    args.config_path = basic_align_config
    with pytest.raises(PronunciationAcousticMismatchError):
        align_included_model(args)

    args.acoustic_model_path = 'english'
    args.corpus_directory = basic_corpus_dir
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = os.path.join(generated_dir, 'basic_output')
    align_included_model(args)

def test_nnet_export_model(basic_corpus_dir, large_prosodylab_format_directory, sick_dict_path, generated_dir, large_dataset_dictionary):
    args = DummyArgs()
    args.artificial_neural_net = True
    args.debug = True
    args.output_model_path = os.path.join(generated_dir, 'nnet_test_model.zip')
    args.corpus_directory = large_prosodylab_format_directory
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = os.path.join(generated_dir, 'nnet_basic_output_selftrained_outputting_model3')
    train_and_align_corpus(args)

def test_align_basic_errors(basic_corpus_dir, large_dataset_dictionary, generated_dir, temp_dir):
    args = DummyArgs()
    args.errors = True
    args.quiet = True
    args.acoustic_model_path = 'english'
    args.corpus_directory = basic_corpus_dir
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = os.path.join(generated_dir, 'basic_output')
    args.temp_directory = temp_dir
    align_included_model(args)


def test_align_basic_debug(basic_corpus_dir, large_dataset_dictionary, generated_dir, temp_dir):
    args = DummyArgs()
    args.debug = True
    args.quiet = True
    args.acoustic_model_path = 'english'
    args.corpus_directory = basic_corpus_dir
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = os.path.join(generated_dir, 'basic_output')
    args.temp_directory = temp_dir
    align_included_model(args)


@large
def test_align_large_prosodylab(large_prosodylab_format_directory, prosodylab_output_directory,
                                large_dataset_dictionary, temp_dir, basic_align_config):
    args = DummyArgs()
    args.quiet = True
    args.acoustic_model_path = 'english'
    args.corpus_directory = large_prosodylab_format_directory
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = prosodylab_output_directory
    args.temp_directory = temp_dir
    args.config_path = basic_align_config
    align_included_model(args)
    # assert_export_exist(large_prosodylab_format_directory, prosodylab_output_directory)


@large
def test_train_large_prosodylab(large_prosodylab_format_directory,
                                large_dataset_dictionary, prosodylab_output_directory,
                                prosodylab_output_model_path, temp_dir, basic_train_config):
    args = DummyArgs()
    args.quiet = True
    args.num_jobs = 2
    args.fast = True
    args.quiet = True
    args.corpus_directory = large_prosodylab_format_directory
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = prosodylab_output_directory
    args.output_model_path = prosodylab_output_model_path
    args.temp_directory = temp_dir
    args.config_path = basic_train_config
    train_and_align_corpus(args)
    # assert_export_exist(large_prosodylab_format_directory, prosodylab_output_directory)
    assert (os.path.exists(args.output_model_path))

    args.clean = False
    train_and_align_corpus(args)


@large
def test_train_single_speaker_prosodylab(single_speaker_prosodylab_format_directory,
                                         large_dataset_dictionary,
                                         prosodylab_output_directory,
                                         prosodylab_output_model_path, temp_dir, basic_train_config):
    args = DummyArgs()
    args.quiet = True
    args.num_jobs = 2
    args.fast = True
    args.quiet = True
    args.temp_directory = temp_dir
    args.corpus_directory = single_speaker_prosodylab_format_directory
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = prosodylab_output_directory
    args.output_model_path = prosodylab_output_model_path
    args.config_path = basic_train_config
    train_and_align_corpus(args)
    # assert_export_exist(single_speaker_prosodylab_format_directory, prosodylab_output_directory)
    assert (os.path.exists(args.output_model_path))


## TEXTGRID

@large
def test_align_large_textgrid(large_textgrid_format_directory, textgrid_output_directory, large_dataset_dictionary, temp_dir, basic_align_config):
    args = DummyArgs()
    args.quiet = True
    args.acoustic_model_path = 'english'
    args.corpus_directory = large_textgrid_format_directory
    args.output_directory = textgrid_output_directory
    args.dictionary_path = large_dataset_dictionary
    args.temp_directory = temp_dir
    args.config_path = basic_align_config
    align_included_model(args)
    # assert_export_exist(large_textgrid_format_directory, textgrid_output_directory)


@large
def test_train_large_textgrid(large_textgrid_format_directory,
                              large_dataset_dictionary, textgrid_output_directory,
                              textgrid_output_model_path, temp_dir, basic_train_config):
    args = DummyArgs()
    args.quiet = True
    args.num_jobs = 2
    args.fast = True
    args.corpus_directory = large_textgrid_format_directory
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = textgrid_output_directory
    args.output_model_path = textgrid_output_model_path
    args.temp_directory = temp_dir
    args.config_path = basic_train_config
    train_and_align_corpus(args)
    # assert_export_exist(large_textgrid_format_directory, textgrid_output_directory)
    assert (os.path.exists(args.output_model_path))


@large
def test_train_large_textgrid_nodict(large_textgrid_format_directory,
                                     textgrid_output_directory,
                                     textgrid_output_model_path, temp_dir, basic_train_config):
    args = DummyArgs()
    args.quiet = True
    args.num_jobs = 2
    args.fast = True
    args.corpus_directory = large_textgrid_format_directory
    args.output_directory = textgrid_output_directory
    args.output_model_path = textgrid_output_model_path
    args.temp_directory = temp_dir
    args.config_path = basic_train_config
    align_corpus_no_dict(args)
    # assert_export_exist(large_textgrid_format_directory, textgrid_output_directory)
    assert (os.path.exists(args.output_model_path))


def test_train_g2p(sick_dict_path, sick_g2p_model_path, temp_dir):
    args = G2PDummyArgs()
    args.validate = True
    args.dictionary_path = sick_dict_path
    args.output_model_path = sick_g2p_model_path
    args.temp_directory = temp_dir
    train_g2p(args)
    assert (os.path.exists(sick_g2p_model_path))


def test_generate_dict(basic_corpus_dir, sick_g2p_model_path, g2p_sick_output, temp_dir):
    args = G2PDummyArgs()
    args.g2p_model_path = sick_g2p_model_path
    args.corpus_directory = basic_corpus_dir
    args.output_path = g2p_sick_output
    args.temp_directory = temp_dir
    generate_dict(args)
    assert (os.path.exists(g2p_sick_output))
