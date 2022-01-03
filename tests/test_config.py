import os

import pytest

from montreal_forced_aligner.acoustic_modeling import (
    LdaTrainer,
    MonophoneTrainer,
    SatTrainer,
    TrainableAligner,
    TriphoneTrainer,
)
from montreal_forced_aligner.alignment import PretrainedAligner
from montreal_forced_aligner.exceptions import ConfigError
from montreal_forced_aligner.ivector.trainer import TrainableIvectorExtractor


def test_monophone_config(sick_corpus, sick_dict, temp_dir):
    am_trainer = TrainableAligner(
        corpus_directory=sick_corpus, dictionary_path=sick_dict, temporary_directory=temp_dir
    )
    config = MonophoneTrainer(identifier="mono", worker=am_trainer)
    config.compute_calculated_properties()
    assert config.realignment_iterations == [
        0,
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
    am_trainer.cleanup()


def test_triphone_config(sick_corpus, sick_dict, temp_dir):
    am_trainer = TrainableAligner(
        corpus_directory=sick_corpus, dictionary_path=sick_dict, temporary_directory=temp_dir
    )
    config = TriphoneTrainer(identifier="tri", worker=am_trainer)
    config.compute_calculated_properties()
    assert config.realignment_iterations == [10, 20, 30]
    am_trainer.cleanup()


def test_lda_mllt_config(sick_corpus, sick_dict, temp_dir):
    am_trainer = TrainableAligner(
        corpus_directory=sick_corpus, dictionary_path=sick_dict, temporary_directory=temp_dir
    )

    assert am_trainer.beam == 10
    assert am_trainer.retry_beam == 40
    assert am_trainer.align_options["beam"] == 10
    assert am_trainer.align_options["retry_beam"] == 40
    config = LdaTrainer(identifier="lda", worker=am_trainer)
    config.compute_calculated_properties()
    assert config.mllt_iterations == [2, 4, 6, 16]
    am_trainer.cleanup()


def test_load_align(
    config_directory,
    sick_corpus,
    sick_dict,
    temp_dir,
    english_acoustic_model,
    mono_align_config_path,
):
    params = PretrainedAligner.parse_parameters(mono_align_config_path)
    aligner = PretrainedAligner(
        acoustic_model_path=english_acoustic_model,
        corpus_directory=sick_corpus,
        dictionary_path=sick_dict,
        temporary_directory=temp_dir,
        **params
    )

    assert params["beam"] == 100
    assert params["retry_beam"] == 400
    assert aligner.beam == 100
    assert aligner.retry_beam == 400
    assert aligner.align_options["beam"] == 100
    assert aligner.align_options["retry_beam"] == 400
    aligner.cleanup()

    path = os.path.join(config_directory, "bad_align_config.yaml")
    params = PretrainedAligner.parse_parameters(path)
    print(params)
    aligner = PretrainedAligner(
        acoustic_model_path=english_acoustic_model,
        corpus_directory=sick_corpus,
        dictionary_path=sick_dict,
        temporary_directory=temp_dir,
        **params
    )
    assert aligner.beam == 10
    assert aligner.retry_beam == 40
    aligner.cleanup()


def test_load_basic_train(sick_corpus, sick_dict, temp_dir, basic_train_config_path):
    params = TrainableAligner.parse_parameters(basic_train_config_path)
    am_trainer = TrainableAligner(
        corpus_directory=sick_corpus,
        dictionary_path=sick_dict,
        temporary_directory=temp_dir,
        **params
    )

    assert am_trainer.beam == 100
    assert am_trainer.retry_beam == 400
    assert am_trainer.align_options["beam"] == 100
    assert am_trainer.align_options["retry_beam"] == 400

    for trainer in am_trainer.training_configs.values():
        assert trainer.beam == 100
        assert trainer.retry_beam == 400
        assert trainer.align_options["beam"] == 100
        assert trainer.align_options["retry_beam"] == 400
    am_trainer.cleanup()


def test_load_mono_train(sick_corpus, sick_dict, temp_dir, mono_train_config_path):
    params = TrainableAligner.parse_parameters(mono_train_config_path)
    am_trainer = TrainableAligner(
        corpus_directory=sick_corpus,
        dictionary_path=sick_dict,
        temporary_directory=temp_dir,
        **params
    )
    for t in am_trainer.training_configs.values():
        assert not t.use_mp
        assert t.use_energy
    assert not am_trainer.use_mp
    assert am_trainer.use_energy
    am_trainer.cleanup()


def test_load_ivector_train(sick_corpus, sick_dict, temp_dir, train_ivector_config_path):
    params = TrainableIvectorExtractor.parse_parameters(train_ivector_config_path)
    trainer = TrainableIvectorExtractor(
        corpus_directory=sick_corpus, temporary_directory=temp_dir, **params
    )

    for t in trainer.training_configs.values():
        assert not t.use_mp
        assert t.use_energy
    assert not trainer.use_mp
    trainer.cleanup()


def test_load(sick_corpus, sick_dict, temp_dir, config_directory):
    path = os.path.join(config_directory, "basic_train_config.yaml")
    params = TrainableAligner.parse_parameters(path)
    am_trainer = TrainableAligner(
        corpus_directory=sick_corpus,
        dictionary_path=sick_dict,
        temporary_directory=temp_dir,
        **params
    )
    assert len(am_trainer.training_configs) == 4
    assert isinstance(am_trainer.training_configs["monophone"], MonophoneTrainer)
    assert isinstance(am_trainer.training_configs["triphone"], TriphoneTrainer)
    assert isinstance(am_trainer.training_configs[am_trainer.final_identifier], SatTrainer)

    path = os.path.join(config_directory, "out_of_order_config.yaml")
    with pytest.raises(ConfigError):
        params = TrainableAligner.parse_parameters(path)
    am_trainer.cleanup()


def test_multilingual_ipa(sick_corpus, sick_dict, temp_dir, config_directory):

    path = os.path.join(config_directory, "basic_ipa_config.yaml")
    params = TrainableAligner.parse_parameters(path)
    am_trainer = TrainableAligner(
        corpus_directory=sick_corpus,
        dictionary_path=sick_dict,
        temporary_directory=temp_dir,
        **params
    )
    am_trainer.cleanup()
