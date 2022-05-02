import argparse
import os
import shutil

from montreal_forced_aligner.acoustic_modeling.trainer import TrainableAligner
from montreal_forced_aligner.alignment import PretrainedAligner


def test_trainer(basic_dict_path, basic_corpus_dir, generated_dir):
    data_directory = os.path.join(generated_dir, "temp", "train_test")
    a = TrainableAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=basic_dict_path,
        temporary_directory=data_directory,
    )
    assert a.final_identifier == "sat_4"
    assert a.training_configs[a.final_identifier].subset == 0
    assert a.training_configs[a.final_identifier].num_leaves == 7000
    assert a.training_configs[a.final_identifier].max_gaussians == 150000


def test_basic_mono(
    basic_dict_path,
    basic_corpus_dir,
    generated_dir,
    mono_train_config_path,
    mono_align_model_path,
    mono_output_directory,
):
    data_directory = os.path.join(generated_dir, "temp", "mono_train_test")
    shutil.rmtree(data_directory, ignore_errors=True)
    args = argparse.Namespace(use_mp=True, debug=False, verbose=True)
    a = TrainableAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=basic_dict_path,
        temporary_directory=data_directory,
        **TrainableAligner.parse_parameters(mono_train_config_path, args=args)
    )
    a.train()
    a.export_model(mono_align_model_path)

    data_directory = os.path.join(generated_dir, "temp", "mono_align_test")
    shutil.rmtree(data_directory, ignore_errors=True)
    a = PretrainedAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=basic_dict_path,
        acoustic_model_path=mono_align_model_path,
        temporary_directory=data_directory,
        **PretrainedAligner.parse_parameters(args=args)
    )
    a.align()
    a.export_files(mono_output_directory)


def test_pronunciation_training(
    basic_dict_path, basic_corpus_dir, generated_dir, pron_train_config_path
):
    data_directory = os.path.join(generated_dir, "temp", "pron_train_test")
    export_path = os.path.join(generated_dir, "pron_train_test_export", "model.zip")
    shutil.rmtree(data_directory, ignore_errors=True)
    args = argparse.Namespace(use_mp=True, debug=False, verbose=True)
    a = TrainableAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=basic_dict_path,
        temporary_directory=data_directory,
        **TrainableAligner.parse_parameters(pron_train_config_path, args=args)
    )
    a.train()

    a.cleanup()
    assert not os.path.exists(export_path)
    assert not os.path.exists(
        os.path.join(generated_dir, "pron_train_test_export", os.path.basename(basic_dict_path))
    )
    a = TrainableAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=basic_dict_path,
        temporary_directory=data_directory,
        **TrainableAligner.parse_parameters(pron_train_config_path, args=args)
    )
    a.train()
    a.export_model(export_path)
    assert os.path.exists(export_path)
    assert os.path.exists(
        os.path.join(
            generated_dir,
            "pron_train_test_export",
            os.path.basename(basic_dict_path).replace(".txt", ".dict"),
        )
    )


def test_basic_tri(basic_dict_path, basic_corpus_dir, generated_dir, tri_train_config_path):
    data_directory = os.path.join(generated_dir, "temp", "tri_test")
    shutil.rmtree(data_directory, ignore_errors=True)
    a = TrainableAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=basic_dict_path,
        temporary_directory=data_directory,
        debug=True,
        verbose=True,
        **TrainableAligner.parse_parameters(tri_train_config_path)
    )
    a.train()


def test_pitch_feature_training(
    basic_dict_path, basic_corpus_dir, generated_dir, pitch_train_config_path
):
    data_directory = os.path.join(generated_dir, "temp", "tri_pitch_test")
    shutil.rmtree(data_directory, ignore_errors=True)
    a = TrainableAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=basic_dict_path,
        temporary_directory=data_directory,
        debug=True,
        verbose=True,
        **TrainableAligner.parse_parameters(pitch_train_config_path)
    )
    assert a.use_pitch
    a.train()
    assert a.get_feat_dim() == 48


def test_basic_lda(basic_dict_path, basic_corpus_dir, generated_dir, lda_train_config_path):
    data_directory = os.path.join(generated_dir, "temp", "lda_test")
    shutil.rmtree(data_directory, ignore_errors=True)
    a = TrainableAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=basic_dict_path,
        temporary_directory=data_directory,
        debug=True,
        verbose=True,
        **TrainableAligner.parse_parameters(lda_train_config_path)
    )
    a.train()
    assert len(a.training_configs[a.final_identifier].realignment_iterations) > 0
    assert len(a.training_configs[a.final_identifier].mllt_iterations) > 1


def test_basic_sat(basic_dict_path, basic_corpus_dir, generated_dir, sat_train_config_path):
    data_directory = os.path.join(generated_dir, "temp", "sat_test")
    output_model_path = os.path.join(data_directory, "sat_model.zip")
    shutil.rmtree(data_directory, ignore_errors=True)
    args = argparse.Namespace(use_mp=True, debug=True, verbose=True)
    a = TrainableAligner(
        **TrainableAligner.parse_parameters(sat_train_config_path, args=args),
        corpus_directory=basic_corpus_dir,
        dictionary_path=basic_dict_path,
        temporary_directory=data_directory,
        disable_mp=False
    )
    a.train()
    assert len(a.training_configs[a.final_identifier].realignment_iterations) > 0
    assert len(a.training_configs[a.final_identifier].fmllr_iterations) > 1
    a.export_model(output_model_path)

    assert os.path.exists(output_model_path)
    assert os.path.exists(
        os.path.join(data_directory, "basic_train_acoustic_model", "sat", "trans.1.0.ark")
    )
