import os
import pytest

from aligner.command_line.align import align_corpus, align_included_model

from aligner.command_line.train_and_align import align_corpus as train_and_align_corpus, align_corpus_no_dict


large = pytest.mark.skipif(
    pytest.config.getoption("--skiplarge"),
    reason="remove --skiplarge option to run"
)

def assert_export_exist(old_directory, new_directory):
    for root, dirs, files in os.walk(old_directory):
        new_root = root.replace(old_directory, new_directory)
        for d in dirs:
            assert(os.path.exists(os.path.join(new_root, d)))
        for f in files:
            if not f.endswith('.wav'):
                continue
            new_f = f.replace('.wav', '.TextGrid')
            assert(os.path.exists(os.path.join(new_root, new_f)))

@large
def test_align_large_prosodylab(large_prosodylab_format_directory, prosodylab_output_directory):
    language = 'english'
    corpus_dir = large_prosodylab_format_directory
    output_directory = prosodylab_output_directory
    speaker_characters = 0
    num_jobs = 0
    verbose = False
    clean = True
    align_included_model(language, corpus_dir,  output_directory,
                            speaker_characters, num_jobs, verbose, clean)
    #assert_export_exist(large_prosodylab_format_directory, prosodylab_output_directory)

@large
def test_train_large_prosodylab(large_prosodylab_format_directory,
                    large_dataset_dictionary, prosodylab_output_directory,
                    prosodylab_output_model_path):
    corpus_dir = large_prosodylab_format_directory
    dict_path = large_dataset_dictionary
    output_directory = prosodylab_output_directory
    output_model_path = prosodylab_output_model_path
    speaker_characters = 0
    num_jobs = 2
    fast = True
    verbose = False
    clean = True
    train_and_align_corpus(corpus_dir, dict_path,  output_directory, speaker_characters, fast,
            output_model_path, num_jobs, verbose, clean)
    #assert_export_exist(large_prosodylab_format_directory, prosodylab_output_directory)
    assert(os.path.exists(output_model_path))

@large
def test_align_single_speaker_prosodylab(single_speaker_prosodylab_format_directory,
                                        large_dataset_dictionary,
                                        prosodylab_output_directory,
                                        prosodylab_output_model_path):
    corpus_dir = single_speaker_prosodylab_format_directory
    dict_path = large_dataset_dictionary
    output_directory = prosodylab_output_directory
    output_model_path = prosodylab_output_model_path
    speaker_characters = 0
    num_jobs = 2
    fast = True
    verbose = False
    clean = True
    train_and_align_corpus(corpus_dir, dict_path,  output_directory, speaker_characters, fast,
            output_model_path, num_jobs, verbose, clean)
    #assert_export_exist(single_speaker_prosodylab_format_directory, prosodylab_output_directory)

## TEXTGRID

@large
def test_align_large_textgrid(large_textgrid_format_directory, textgrid_output_directory):
    language = 'english'
    corpus_dir = large_textgrid_format_directory
    output_directory = textgrid_output_directory
    speaker_characters = 0
    num_jobs = 0
    verbose = False
    clean = True
    align_included_model(language, corpus_dir,  output_directory,
                            speaker_characters, num_jobs, verbose, clean)
    #assert_export_exist(large_textgrid_format_directory, textgrid_output_directory)

@large
def test_train_large_textgrid(large_textgrid_format_directory,
                    large_dataset_dictionary, textgrid_output_directory,
                    textgrid_output_model_path):
    corpus_dir = large_textgrid_format_directory
    dict_path = large_dataset_dictionary
    output_directory = textgrid_output_directory
    output_model_path = textgrid_output_model_path
    speaker_characters = 0
    num_jobs = 2
    fast = True
    verbose = False
    clean = True
    train_and_align_corpus(corpus_dir, dict_path,  output_directory, speaker_characters, fast,
            output_model_path, num_jobs, verbose, clean)
    #assert_export_exist(large_textgrid_format_directory, textgrid_output_directory)
    assert(os.path.exists(output_model_path))

@large
def test_train_large_textgrid_nodict(large_textgrid_format_directory,
                    textgrid_output_directory,
                    textgrid_output_model_path):
    corpus_dir = large_textgrid_format_directory
    output_directory = textgrid_output_directory
    output_model_path = textgrid_output_model_path
    speaker_characters = 0
    num_jobs = 2
    fast = True
    verbose = True
    clean = True
    align_corpus_no_dict(corpus_dir,  output_directory, speaker_characters, fast,
            output_model_path, num_jobs, verbose, clean)
    #assert_export_exist(large_textgrid_format_directory, textgrid_output_directory)
    assert(os.path.exists(output_model_path))
