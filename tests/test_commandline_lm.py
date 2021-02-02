import os
import pytest
import sys

from montreal_forced_aligner.command_line.train_lm import run_train_lm
from montreal_forced_aligner.command_line.mfa import parser


def test_train_lm(basic_corpus_dir, temp_dir, generated_dir, basic_train_lm_config):
    if sys.platform == 'win32':
        pytest.skip('LM training not supported on Windows.')
    command = ['train_lm', basic_corpus_dir, os.path.join(generated_dir, 'test_basic_lm.zip'),
               '-t', temp_dir, '-c', basic_train_lm_config, '-q', '--clean']
    args, unknown = parser.parse_known_args(command)
    run_train_lm(args)
    assert os.path.exists(args.output_model_path)
