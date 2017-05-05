import pytest
import os
from aligner.g2p_trainer.train import Trainer

def test_training(dict_model_path, dict_output_path):
    path_to_model = Trainer(dict_model_path, dict_output_path, **{}).get_path_to_model()
    assert(os.path.split(path_to_model)[0] == dict_model_path)



def test_training_CH_chars(dict_model_path_char, dict_output_path_char):
    path_to_model = Trainer(dict_model_path_char, dict_output_path_char, **{"CH_chars":True}).get_path_to_model()
    assert(os.path.split(path_to_model)[0] == dict_model_path_char)

