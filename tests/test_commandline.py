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
        self.fast = True
        self.no_speaker_adaptation = False
        self.debug = False
        self.errors = False
        self.temp_directory = None


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


def test_align_basic(basic_corpus_dir, sick_dict_path, generated_dir, large_dataset_dictionary):
    args = DummyArgs()
    args.acoustic_model_path = 'english'
    args.corpus_directory = basic_corpus_dir
    args.dictionary_path = sick_dict_path
    args.output_directory = os.path.join(generated_dir, 'basic_output')
    with pytest.raises(PronunciationAcousticMismatchError):
        align_included_model(args, skip_input=True)

    #args.clean = False
    #args.acoustic_model_path = 'english'
    #align_included_model(args, skip_input=True)
    args.acoustic_model_path = 'english'
    args.corpus_directory = basic_corpus_dir
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = os.path.join(generated_dir, 'basic_output')
    align_included_model(args, skip_input=True)


def test_align_basic_errors(basic_corpus_dir, large_dataset_dictionary, generated_dir):
    args = DummyArgs()
    args.errors = True
    args.acoustic_model_path = 'english'
    args.corpus_directory = basic_corpus_dir
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = os.path.join(generated_dir, 'basic_output')
    align_included_model(args, skip_input=True)


def test_align_basic_debug(basic_corpus_dir, large_dataset_dictionary, generated_dir):
    args = DummyArgs()
    args.debug = True
    args.acoustic_model_path = 'english'
    args.corpus_directory = basic_corpus_dir
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = os.path.join(generated_dir, 'basic_output')
    align_included_model(args, skip_input=True)


@large
def test_align_large_prosodylab(large_prosodylab_format_directory, prosodylab_output_directory,
                                large_dataset_dictionary):
    args = DummyArgs()
    args.acoustic_model_path = 'english'
    args.corpus_directory = large_prosodylab_format_directory
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = prosodylab_output_directory
    align_included_model(args, skip_input=True)
    # assert_export_exist(large_prosodylab_format_directory, prosodylab_output_directory)


@large
def test_train_large_prosodylab(large_prosodylab_format_directory,
                                large_dataset_dictionary, prosodylab_output_directory,
                                prosodylab_output_model_path, temp_dir):
    args = DummyArgs()
    args.num_jobs = 2
    args.fast = True
    args.corpus_directory = large_prosodylab_format_directory
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = prosodylab_output_directory
    args.output_model_path = prosodylab_output_model_path
    args.temp_directory = temp_dir
    train_and_align_corpus(args, skip_input=True)
    # assert_export_exist(large_prosodylab_format_directory, prosodylab_output_directory)
    assert (os.path.exists(args.output_model_path))

    args.clean = False
    train_and_align_corpus(args, skip_input=True)


@large
def test_train_single_speaker_prosodylab(single_speaker_prosodylab_format_directory,
                                         large_dataset_dictionary,
                                         prosodylab_output_directory,
                                         prosodylab_output_model_path):
    args = DummyArgs()
    args.num_jobs = 2
    args.fast = True
    args.corpus_directory = single_speaker_prosodylab_format_directory
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = prosodylab_output_directory
    args.output_model_path = prosodylab_output_model_path
    train_and_align_corpus(args, skip_input=True)
    # assert_export_exist(single_speaker_prosodylab_format_directory, prosodylab_output_directory)
    assert (os.path.exists(args.output_model_path))


## TEXTGRID

@large
def test_align_large_textgrid(large_textgrid_format_directory, textgrid_output_directory, large_dataset_dictionary):
    args = DummyArgs()
    args.acoustic_model_path = 'english'
    args.corpus_directory = large_textgrid_format_directory
    args.output_directory = textgrid_output_directory
    args.dictionary_path = large_dataset_dictionary
    align_included_model(args, skip_input=True)
    # assert_export_exist(large_textgrid_format_directory, textgrid_output_directory)


@large
def test_train_large_textgrid(large_textgrid_format_directory,
                              large_dataset_dictionary, textgrid_output_directory,
                              textgrid_output_model_path):
    args = DummyArgs()
    args.num_jobs = 2
    args.fast = True
    args.corpus_directory = large_textgrid_format_directory
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = textgrid_output_directory
    args.output_model_path = textgrid_output_model_path
    train_and_align_corpus(args, skip_input=True)
    # assert_export_exist(large_textgrid_format_directory, textgrid_output_directory)
    assert (os.path.exists(args.output_model_path))


@large
def test_train_large_textgrid_nodict(large_textgrid_format_directory,
                                     textgrid_output_directory,
                                     textgrid_output_model_path):
    args = DummyArgs()
    args.num_jobs = 2
    args.fast = True
    args.corpus_directory = large_textgrid_format_directory
    args.output_directory = textgrid_output_directory
    args.output_model_path = textgrid_output_model_path
    align_corpus_no_dict(args, skip_input=True)
    # assert_export_exist(large_textgrid_format_directory, textgrid_output_directory)
    assert (os.path.exists(args.output_model_path))


def test_train_g2p(sick_dict_path, sick_g2p_model_path):
    args = G2PDummyArgs()
    args.dictionary_path = sick_dict_path
    args.output_model_path = sick_g2p_model_path
    train_g2p(args)
    assert (os.path.exists(sick_g2p_model_path))


def test_generate_dict(basic_corpus_dir, sick_g2p_model_path, g2p_sick_output):
    args = G2PDummyArgs()
    args.g2p_model_path = sick_g2p_model_path
    args.corpus_directory = basic_corpus_dir
    args.output_path = g2p_sick_output
    generate_dict(args)
    assert (os.path.exists(g2p_sick_output))