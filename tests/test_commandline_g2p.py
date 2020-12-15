import os
import pytest

from montreal_forced_aligner.g2p.generator import G2P_DISABLED
from montreal_forced_aligner.command_line.g2p import run_g2p
from montreal_forced_aligner.command_line.train_g2p import run_train_g2p

from montreal_forced_aligner.dictionary import Dictionary


class G2PDummyArgs(object):
    def __init__(self):
        self.temp_directory = None
        self.window_size = 2
        self.order = 5
        self.random_starts=1
        self.num_jobs = 2
        self.disable_mp = False
        self.include_bracketed = False


def test_train_g2p(sick_dict_path, sick_g2p_model_path, temp_dir):
    if G2P_DISABLED:
        pytest.skip('No Pynini found')
    args = G2PDummyArgs()
    args.validate = True
    args.dictionary_path = sick_dict_path
    args.output_model_path = sick_g2p_model_path
    args.temp_directory = temp_dir
    run_train_g2p(args)
    assert os.path.exists(sick_g2p_model_path)


def test_generate_dict(basic_corpus_dir, sick_g2p_model_path, g2p_sick_output, temp_dir):
    if G2P_DISABLED:
        pytest.skip('No Pynini found')
    args = G2PDummyArgs()
    args.g2p_model_path = sick_g2p_model_path
    args.input_path = basic_corpus_dir
    args.output_path = g2p_sick_output
    args.temp_directory = temp_dir
    run_g2p(args)
    assert os.path.exists(g2p_sick_output)
    d = Dictionary(g2p_sick_output, temp_dir)
    assert len(d.words) > 0


def test_generate_orthography_dict(basic_corpus_dir, orth_sick_output, temp_dir):
    if G2P_DISABLED:
        pytest.skip('No Pynini found')
    args = G2PDummyArgs()
    args.g2p_model_path = None
    args.input_path = basic_corpus_dir
    args.output_path = orth_sick_output
    args.temp_directory = temp_dir
    run_g2p(args)
    assert os.path.exists(orth_sick_output)
    d = Dictionary(orth_sick_output, temp_dir)
    assert len(d.words) > 0