import os
import pytest
from aligner.config import FeatureConfig, train_yaml_to_config, ConfigError
from aligner.trainers import MonophoneTrainer, TriphoneTrainer, LdaTrainer, SatTrainer

def test_monophone_config():
    config = MonophoneTrainer(FeatureConfig())
    assert config.realignment_iterations == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14,
                                  16, 18, 20, 23, 26, 29, 32, 35, 38]


def test_triphone_config():
    config = TriphoneTrainer(FeatureConfig())
    assert config.realignment_iterations == [10, 20, 30]


def test_lda_mllt_config():
    config = LdaTrainer(FeatureConfig())
    assert config.mllt_iterations == [2, 4, 6, 16]


def test_load(config_directory):
    path = os.path.join(config_directory, 'basic_train_config.yaml')
    train, align = train_yaml_to_config(path)
    assert len(train.training_configs) == 4
    assert isinstance(train.training_configs[0], MonophoneTrainer)
    assert isinstance(train.training_configs[1], TriphoneTrainer)
    assert isinstance(train.training_configs[-1], SatTrainer)

    path = os.path.join(config_directory, 'out_of_order_config.yaml')
    with pytest.raises(ConfigError):
        train, align = train_yaml_to_config(path)