import os
import pytest

from montreal_forced_aligner.g2p.generator import G2P_DISABLED
from montreal_forced_aligner.command_line.g2p import run_g2p
from montreal_forced_aligner.command_line.train_g2p import run_train_g2p
from montreal_forced_aligner.command_line.mfa import parser

from montreal_forced_aligner.dictionary import Dictionary


def test_train_g2p(sick_dict_path, sick_g2p_model_path, temp_dir):
    if G2P_DISABLED:
        pytest.skip('No Pynini found')
    command = ['train_g2p', sick_dict_path, sick_g2p_model_path,
               '-t', temp_dir, '-q', '--clean', '-d', '--validate']
    args, unknown = parser.parse_known_args(command)
    run_train_g2p(args)
    assert os.path.exists(sick_g2p_model_path)


def test_generate_dict(basic_corpus_dir, sick_g2p_model_path, g2p_sick_output, temp_dir):
    if G2P_DISABLED:
        pytest.skip('No Pynini found')
    command = ['g2p', sick_g2p_model_path, basic_corpus_dir, g2p_sick_output,
               '-t', temp_dir, '-q', '--clean', '-d']
    args, unknown = parser.parse_known_args(command)
    run_g2p(args)
    assert os.path.exists(g2p_sick_output)
    d = Dictionary(g2p_sick_output, temp_dir)
    assert len(d.words) > 0


def test_generate_orthography_dict(basic_corpus_dir, orth_sick_output, temp_dir):
    if G2P_DISABLED:
        pytest.skip('No Pynini found')
    command = ['g2p', basic_corpus_dir, orth_sick_output,
               '-t', temp_dir, '-q', '--clean', '-d']
    args, unknown = parser.parse_known_args(command)
    run_g2p(args)
    assert os.path.exists(orth_sick_output)
    d = Dictionary(orth_sick_output, temp_dir)
    assert len(d.words) > 0