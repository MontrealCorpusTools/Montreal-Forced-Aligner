import pytest
import os
from aligner.g2p.trainer import PhonetisaurusTrainer

from aligner.g2p.generator import PhonetisaurusDictionaryGenerator

from aligner.models import G2PModel

from aligner import __version__


def test_training(sick_dict, sick_g2p_model_path):
    trainer = PhonetisaurusTrainer(sick_dict, sick_g2p_model_path, window_size=2)
    trainer.validate()
    trainer.train()
    model = G2PModel(sick_g2p_model_path)
    assert model.meta['version'] == __version__
    assert model.meta['architecture'] == 'phonetisaurus'
    assert model.meta['phones'] == sick_dict.nonsil_phones


def test_generator(sick_g2p_model_path, sick_corpus, g2p_sick_output):
    model = G2PModel(sick_g2p_model_path)
    gen = PhonetisaurusDictionaryGenerator(model, sick_corpus.word_set, g2p_sick_output)
    gen.generate()
    assert os.path.exists(g2p_sick_output)
