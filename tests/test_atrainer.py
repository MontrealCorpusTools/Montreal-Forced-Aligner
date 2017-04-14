import pytest
import os
from aligner.g2p_trainer.train import Trainer

def test_training(dict_model_path, dict_output_path):
    path_to_model = Trainer(dict_model_path, dict_output_path, False).get_path_to_model()
    assert(os.path.split(path_to_model)[0] == dict_model_path)
