import os
import pytest

from montreal_forced_aligner.command_line.align import run_align_corpus, DummyArgs

from montreal_forced_aligner.command_line.train_and_align import run_train_corpus
from montreal_forced_aligner.command_line.validate import validate_corpus

from montreal_forced_aligner.exceptions import PronunciationAcousticMismatchError


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



def test_align_basic(basic_corpus_dir, sick_dict_path, generated_dir, large_dataset_dictionary, temp_dir,
                     basic_align_config, english_acoustic_model):
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
        run_align_corpus(args)

    args.acoustic_model_path = 'english'
    args.corpus_directory = basic_corpus_dir
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = os.path.join(generated_dir, 'basic_output')
    run_align_corpus(args)


def test_align_basic_errors(basic_corpus_dir, large_dataset_dictionary, generated_dir, temp_dir):
    args = DummyArgs()
    args.quiet = True
    args.clean = True
    args.acoustic_model_path = 'english'
    args.corpus_directory = basic_corpus_dir
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = os.path.join(generated_dir, 'basic_output')
    args.temp_directory = temp_dir
    run_align_corpus(args)


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
    run_align_corpus(args)

