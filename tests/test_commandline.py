import os
import pytest

from aligner.command_line.align import align_corpus, align_included_model

from aligner.command_line.train_and_align import align_corpus as train_and_align_corpus
from aligner.command_line.validate_dataset import validate_corpus

from aligner.exceptions import PronunciationAcousticMismatchError


class DummyArgs(object):
    def __init__(self):
        self.speaker_characters = 0
        self.num_jobs = 0
        self.verbose = False
        self.clean = True
        self.debug = False
        self.temp_directory = None
        self.config_path = ''



class ValidatorDummyArgs(object):
    def __init__(self):
        self.temp_directory = None
        self.test_transcriptions = False
        self.num_jobs = 0
        self.speaker_characters = 0




large = pytest.mark.skipif(
    pytest.config.getoption("--skiplarge"),
    reason="remove --skiplarge option to run"
)


def test_validate_corpus(large_prosodylab_format_directory, large_dataset_dictionary,temp_dir):
    args = ValidatorDummyArgs()
    args.num_jobs = 2
    args.corpus_directory = large_prosodylab_format_directory
    args.dictionary_path = large_dataset_dictionary
    args.temp_directory = temp_dir
    args.test_transcriptions = True
    validate_corpus(args)


@large
def test_train_large_prosodylab(large_prosodylab_format_directory,
                                large_dataset_dictionary, prosodylab_output_directory,
                                prosodylab_output_model_path, temp_dir, basic_train_config):
    args = DummyArgs()
    args.num_jobs = 2
    args.clean = True
    args.corpus_directory = large_prosodylab_format_directory
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = prosodylab_output_directory
    args.output_model_path = prosodylab_output_model_path
    args.temp_directory = temp_dir
    args.config_path = basic_train_config
    train_and_align_corpus(args)
    # assert_export_exist(large_prosodylab_format_directory, prosodylab_output_directory)
    assert (os.path.exists(args.output_model_path))

    train_and_align_corpus(args)

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
    args.clean = True
    args.temp_directory = temp_dir
    args.config_path = basic_align_config
    with pytest.raises(PronunciationAcousticMismatchError):
        align_included_model(args)

    args.acoustic_model_path = 'english'
    args.corpus_directory = basic_corpus_dir
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = os.path.join(generated_dir, 'basic_output')
    align_included_model(args)

def test_nnet_export_model(large_prosodylab_format_directory, config_directory, generated_dir, large_dataset_dictionary, temp_dir):
    args = DummyArgs()
    args.debug = True
    args.clean = True
    args.config_path = os.path.join(config_directory, 'long_nnet_train.yaml')
    args.output_model_path = os.path.join(generated_dir, 'nnet_test_model.zip')
    args.corpus_directory = large_prosodylab_format_directory
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = os.path.join(generated_dir, 'nnet_basic_output_selftrained_outputting_model3')
    args.temp_directory = temp_dir
    train_and_align_corpus(args)

def test_nnet_use_model(basic_corpus_dir, generated_dir, large_dataset_dictionary, temp_dir):
    args = DummyArgs()
    args.debug = True
    args.clean = True
    args.acoustic_model_path = os.path.join(generated_dir, 'nnet_test_model.zip')
    args.corpus_directory = basic_corpus_dir
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = os.path.join(generated_dir, 'nnet_basic_output')
    args.temp_directory = temp_dir
    align_corpus(args)

def test_align_basic_errors(basic_corpus_dir, large_dataset_dictionary, generated_dir, temp_dir):
    args = DummyArgs()
    args.quiet = True
    args.clean = True
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
    args.clean = True
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
    args.clean = True
    args.acoustic_model_path = 'english'
    args.corpus_directory = large_prosodylab_format_directory
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = prosodylab_output_directory
    args.temp_directory = temp_dir
    args.config_path = basic_align_config
    align_included_model(args)
    # assert_export_exist(large_prosodylab_format_directory, prosodylab_output_directory)



@large
def test_train_single_speaker_prosodylab(single_speaker_prosodylab_format_directory,
                                         large_dataset_dictionary,
                                         prosodylab_output_directory,
                                         prosodylab_output_model_path, temp_dir, basic_train_config):
    args = DummyArgs()
    args.num_jobs = 2
    args.clean = True
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
    args.clean = True
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
    args.num_jobs = 2
    args.clean = True
    args.corpus_directory = large_textgrid_format_directory
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = textgrid_output_directory
    args.output_model_path = textgrid_output_model_path
    args.temp_directory = temp_dir
    args.config_path = basic_train_config
    train_and_align_corpus(args)
    # assert_export_exist(large_textgrid_format_directory, textgrid_output_directory)
    assert (os.path.exists(args.output_model_path))
