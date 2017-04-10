from aligner.g2p_trainer.train import Trainer

def test_training(dict_language, input_dict):
    path_to_model = Trainer(language, input_dict)