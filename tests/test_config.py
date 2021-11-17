import os

import pytest

from montreal_forced_aligner.config import (
    FeatureConfig,
    align_yaml_to_config,
    train_yaml_to_config,
)
from montreal_forced_aligner.exceptions import ConfigError
from montreal_forced_aligner.trainers import (
    LdaTrainer,
    MonophoneTrainer,
    SatTrainer,
    TriphoneTrainer,
)


def test_monophone_config():
    config = MonophoneTrainer(FeatureConfig())
    assert config.realignment_iterations == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        12,
        14,
        16,
        18,
        20,
        23,
        26,
        29,
        32,
        35,
        38,
    ]


def test_triphone_config():
    config = TriphoneTrainer(FeatureConfig())
    assert config.realignment_iterations == [10, 20, 30]


def test_lda_mllt_config():
    config = LdaTrainer(FeatureConfig())
    assert config.mllt_iterations == [2, 4, 6, 16]


def test_load_align(config_directory, mono_align_config_path):
    _ = align_yaml_to_config(mono_align_config_path)

    path = os.path.join(config_directory, "bad_align_config.yaml")
    with pytest.raises(ConfigError):
        _ = align_yaml_to_config(path)


def test_load_basic_train(config_directory, basic_train_config):
    training_config, align_config, dictioanry_config = train_yaml_to_config(basic_train_config)
    assert align_config.beam == 100
    assert align_config.retry_beam == 400
    assert align_config.align_options["beam"] == 100
    assert align_config.align_options["retry_beam"] == 400

    for trainer in training_config.training_configs:
        assert trainer.beam == 100
        assert trainer.retry_beam == 400
        assert trainer.align_options["beam"] == 100
        assert trainer.align_options["retry_beam"] == 400


def test_load_mono_train(config_directory, mono_train_config_path):
    train, align, dictioanry_config = train_yaml_to_config(mono_train_config_path)
    for t in train.training_configs:
        assert not t.use_mp
        assert not t.feature_config.use_mp
        assert t.feature_config.use_energy
    assert not align.use_mp
    assert not align.feature_config.use_mp
    assert align.feature_config.use_energy


def test_load_ivector_train(config_directory, train_ivector_config):
    train, align, dictioanry_config = train_yaml_to_config(train_ivector_config)
    for t in train.training_configs:
        assert not t.use_mp
        assert not t.feature_config.use_mp
        assert t.feature_config.use_energy
    assert not align.use_mp
    assert not align.feature_config.use_mp


def test_load(config_directory):
    path = os.path.join(config_directory, "basic_train_config.yaml")
    train, align, dictionary_config = train_yaml_to_config(path)
    assert len(train.training_configs) == 4
    assert isinstance(train.training_configs[0], MonophoneTrainer)
    assert isinstance(train.training_configs[1], TriphoneTrainer)
    assert isinstance(train.training_configs[-1], SatTrainer)

    path = os.path.join(config_directory, "out_of_order_config.yaml")
    with pytest.raises(ConfigError):
        train, align, dictionary_config = train_yaml_to_config(path)


def test_multilingual_ipa(config_directory):
    from montreal_forced_aligner.config.dictionary_config import DEFAULT_STRIP_DIACRITICS

    path = os.path.join(config_directory, "basic_ipa_config.yaml")
    train, align, dictionary_config = train_yaml_to_config(path)
    assert dictionary_config.multilingual_ipa
    assert set(dictionary_config.strip_diacritics) == set(DEFAULT_STRIP_DIACRITICS)
    assert dictionary_config.digraphs == ["[dt][szʒʃʐʑʂɕç]", "[a][job_name][u]"]
