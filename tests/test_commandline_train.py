import os
import pytest
from montreal_forced_aligner.command_line.train_and_align import run_train_corpus
from montreal_forced_aligner.command_line.mfa import parser


# @pytest.mark.skip(reason='Optimization')
def test_train_and_align_basic(basic_corpus_dir, sick_dict_path, generated_dir, temp_dir,
                               mono_train_config_path, textgrid_output_model_path):
    if os.path.exists(textgrid_output_model_path):
        os.remove(textgrid_output_model_path)
    command = ['train', basic_corpus_dir, sick_dict_path, os.path.join(generated_dir, 'basic_output'),
               '-t', temp_dir, '--config_path', mono_train_config_path, '-q', '--clean', '-d', '-o', textgrid_output_model_path]
    args, unknown = parser.parse_known_args(command)
    run_train_corpus(args, unknown)
    assert os.path.exists(textgrid_output_model_path)

@pytest.mark.skip(reason='Optimization')
def test_train_and_align_basic_speaker_dict(basic_corpus_dir, speaker_dictionary_path, generated_dir, temp_dir,
                               mono_train_config_path, textgrid_output_model_path):
    if os.path.exists(textgrid_output_model_path):
        os.remove(textgrid_output_model_path)
    command = ['train', basic_corpus_dir, speaker_dictionary_path, os.path.join(generated_dir, 'basic_output'),
               '-t', temp_dir, '--config_path', mono_train_config_path, '-q', '--clean', '-d', '-o', textgrid_output_model_path]
    args, unknown = parser.parse_known_args(command)
    run_train_corpus(args, unknown)
    assert os.path.exists(textgrid_output_model_path)
