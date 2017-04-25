import pytest
import os
from aligner.g2p.train import Trainer

from aligner.g2p.create_dictionary import DictMaker


def test_training(training_dict_path, g2p_model_path):
    trainer = Trainer(training_dict_path, g2p_model_path, korean=False)


def test_creation(dict_model_path, dict_input_directory, dict_output_path):
    D = DictMaker(dict_model_path, dict_input_directory, dict_output_path)
    assert (os.path.exists(dict_output_path))
