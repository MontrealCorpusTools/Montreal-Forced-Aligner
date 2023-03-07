import shutil
import time

import pytest
import sqlalchemy.orm

from montreal_forced_aligner.acoustic_modeling.trainer import TrainableAligner
from montreal_forced_aligner.alignment import PretrainedAligner
from montreal_forced_aligner.db import PhonologicalRule


def test_trainer(basic_dict_path, temp_dir, basic_corpus_dir):
    a = TrainableAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=basic_dict_path,
    )
    assert a.final_identifier == "sat_4"
    assert a.training_configs[a.final_identifier].subset == 0
    assert a.training_configs[a.final_identifier].num_leaves == 7000
    assert a.training_configs[a.final_identifier].max_gaussians == 150000


def test_basic_mono(
    mixed_dict_path,
    basic_corpus_dir,
    mono_train_config_path,
    mono_align_config_path,
    mono_align_model_path,
    mono_output_directory,
    db_setup,
):
    a = TrainableAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=mixed_dict_path,
        **TrainableAligner.parse_parameters(mono_train_config_path)
    )
    a.train()
    a.export_model(mono_align_model_path)
    assert mono_align_model_path.exists()
    a.cleanup()
    a.clean_working_directory()
    del a
    sqlalchemy.orm.close_all_sessions()
    time.sleep(3)
    a = PretrainedAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=mixed_dict_path,
        acoustic_model_path=mono_align_model_path,
        **PretrainedAligner.parse_parameters(mono_align_config_path)
    )
    a.align()
    a.export_files(mono_output_directory)
    assert mono_output_directory.exists()
    a.clean_working_directory()


def test_pronunciation_training(
    mixed_dict_path,
    basic_corpus_dir,
    generated_dir,
    pron_train_config_path,
    rules_path,
    groups_path,
    db_setup,
):
    export_path = generated_dir.joinpath("pron_train_test_export", "model.zip")
    a = TrainableAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=mixed_dict_path,
        rules_path=rules_path,
        phone_groups_path=groups_path,
        **TrainableAligner.parse_parameters(pron_train_config_path)
    )
    a.train()
    assert "coronal_fricatives" in a.phone_groups
    assert set(a.phone_groups["coronal_fricatives"]) == {"s", "z", "sh"}
    with a.session() as session:
        assert session.query(PhonologicalRule).count() > 0
        rule_query = session.query(PhonologicalRule).first()
        assert rule_query.probability > 0
        assert rule_query.probability < 1

    a.clean_working_directory()
    assert not export_path.exists()
    assert not (generated_dir.joinpath("pron_train_test_export", mixed_dict_path.name).exists())

    sqlalchemy.orm.close_all_sessions()
    a = TrainableAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=mixed_dict_path,
        **TrainableAligner.parse_parameters(pron_train_config_path)
    )
    a.train()
    a.export_model(export_path)
    assert export_path.exists()
    assert generated_dir.joinpath(
        "pron_train_test_export", mixed_dict_path.with_suffix(".dict").name
    ).exists()
    a.clean_working_directory()


def test_pitch_feature_training(
    basic_dict_path, basic_corpus_dir, pitch_train_config_path, db_setup
):
    a = TrainableAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=basic_dict_path,
        debug=True,
        verbose=True,
        **TrainableAligner.parse_parameters(pitch_train_config_path)
    )
    assert a.use_pitch
    a.train()
    assert a.get_feat_dim() == 45
    a.clean_working_directory()


def test_basic_lda(basic_dict_path, basic_corpus_dir, lda_train_config_path, db_setup):
    a = TrainableAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=basic_dict_path,
        debug=True,
        verbose=True,
        **TrainableAligner.parse_parameters(lda_train_config_path)
    )
    a.train()
    assert len(a.training_configs[a.final_identifier].realignment_iterations) > 0
    assert len(a.training_configs[a.final_identifier].mllt_iterations) > 1
    a.clean_working_directory()


@pytest.mark.skip("Inconsistent failing")
def test_basic_sat(
    basic_dict_path, basic_corpus_dir, generated_dir, sat_train_config_path, db_setup
):
    data_directory = generated_dir.joinpath("sat_test")
    output_model_path = data_directory.joinpath("sat_model.zip")
    shutil.rmtree(data_directory, ignore_errors=True)
    a = TrainableAligner(
        **TrainableAligner.parse_parameters(sat_train_config_path),
        corpus_directory=basic_corpus_dir,
        dictionary_path=basic_dict_path,
        disable_mp=False
    )
    a.train()
    assert len(a.training_configs[a.final_identifier].fmllr_iterations) > 1
    a.export_model(output_model_path)

    assert output_model_path.exists()
    assert a.output_directory.joinpath("sat", "trans.1.1.ark").exists()
    a.clean_working_directory()
