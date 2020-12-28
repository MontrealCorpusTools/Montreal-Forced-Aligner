import os
import pytest

from montreal_forced_aligner.command_line.train_lm import run_train_lm


class DummyArgs(object):
    def __init__(self):
        self.corpus_directory = ''
        self.output_model_path = ''
        self.dictionary_path = ''
        self.speaker_characters = 0
        self.dictionary = ''
        self.config_path = ''
        self.model_path = ''
        self.num_jobs = 0
        self.verbose = False
        self.clean = True
        self.fast = True
        self.debug = False
        self.temp_directory = None


def test_train_lm(basic_corpus_dir, temp_dir, generated_dir):
    args = DummyArgs()
    args.corpus_directory = basic_corpus_dir
    args.temp_directory = temp_dir
    args.output_model_path = os.path.join(generated_dir, 'test_basic_lm.arpa')
    run_train_lm(args)