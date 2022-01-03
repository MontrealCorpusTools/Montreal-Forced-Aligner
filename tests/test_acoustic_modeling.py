import argparse
import os
import shutil

from montreal_forced_aligner.acoustic_modeling.trainer import TrainableAligner
from montreal_forced_aligner.alignment import PretrainedAligner


def test_trainer(sick_dict, sick_corpus, generated_dir):
    data_directory = os.path.join(generated_dir, "temp", "train_test")
    a = TrainableAligner(
        corpus_directory=sick_corpus,
        dictionary_path=sick_dict,
        temporary_directory=data_directory,
    )
    assert a.final_identifier == "sat_2"
    assert a.training_configs[a.final_identifier].subset == 0
    assert a.training_configs[a.final_identifier].num_leaves == 4200
    assert a.training_configs[a.final_identifier].max_gaussians == 40000


def test_sick_mono(
    sick_dict,
    sick_corpus,
    generated_dir,
    mono_train_config_path,
    mono_align_model_path,
    mono_output_directory,
):
    data_directory = os.path.join(generated_dir, "temp", "mono_train_test")
    shutil.rmtree(data_directory, ignore_errors=True)
    args = argparse.Namespace(use_mp=True, debug=False, verbose=True)
    a = TrainableAligner(
        corpus_directory=sick_corpus,
        dictionary_path=sick_dict,
        temporary_directory=data_directory,
        **TrainableAligner.parse_parameters(mono_train_config_path, args=args)
    )
    a.train()
    a.export_model(mono_align_model_path)

    data_directory = os.path.join(generated_dir, "temp", "mono_align_test")
    shutil.rmtree(data_directory, ignore_errors=True)
    a = PretrainedAligner(
        corpus_directory=sick_corpus,
        dictionary_path=sick_dict,
        acoustic_model_path=mono_align_model_path,
        temporary_directory=data_directory,
        **PretrainedAligner.parse_parameters(args=args)
    )
    a.align()
    a.export_files(mono_output_directory)


def test_sick_tri(sick_dict, sick_corpus, generated_dir, tri_train_config_path):
    data_directory = os.path.join(generated_dir, "temp", "tri_test")
    shutil.rmtree(data_directory, ignore_errors=True)
    a = TrainableAligner(
        corpus_directory=sick_corpus,
        dictionary_path=sick_dict,
        temporary_directory=data_directory,
        debug=True,
        verbose=True,
        **TrainableAligner.parse_parameters(tri_train_config_path)
    )
    a.train()


def test_sick_lda(sick_dict, sick_corpus, generated_dir, lda_train_config_path):
    data_directory = os.path.join(generated_dir, "temp", "lda_test")
    shutil.rmtree(data_directory, ignore_errors=True)
    a = TrainableAligner(
        corpus_directory=sick_corpus,
        dictionary_path=sick_dict,
        temporary_directory=data_directory,
        debug=True,
        verbose=True,
        **TrainableAligner.parse_parameters(lda_train_config_path)
    )
    a.train()
    assert len(a.training_configs[a.final_identifier].realignment_iterations) > 0
    assert len(a.training_configs[a.final_identifier].mllt_iterations) > 1


def test_sick_sat(sick_dict, sick_corpus, generated_dir, sat_train_config_path):
    data_directory = os.path.join(generated_dir, "temp", "sat_test")
    output_model_path = os.path.join(data_directory, "sat_model.zip")
    shutil.rmtree(data_directory, ignore_errors=True)
    args = argparse.Namespace(use_mp=True, debug=True, verbose=True)
    a = TrainableAligner(
        **TrainableAligner.parse_parameters(sat_train_config_path, args=args),
        corpus_directory=sick_corpus,
        dictionary_path=sick_dict,
        temporary_directory=data_directory,
        disable_mp=False
    )
    a.train()
    assert len(a.training_configs[a.final_identifier].realignment_iterations) > 0
    assert len(a.training_configs[a.final_identifier].fmllr_iterations) > 1
    a.export_model(output_model_path)

    assert os.path.exists(output_model_path)
    assert os.path.exists(
        os.path.join(data_directory, "basic_train_acoustic_model", "sat", "trans.sick.0.ark")
    )
