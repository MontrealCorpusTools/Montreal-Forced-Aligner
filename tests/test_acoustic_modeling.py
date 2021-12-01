import os
import shutil

from montreal_forced_aligner.acoustic_modeling.trainer import TrainableAligner
from montreal_forced_aligner.alignment import PretrainedAligner


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
    a = TrainableAligner(
        corpus_directory=sick_corpus,
        dictionary_path=sick_dict,
        temporary_directory=data_directory,
        debug=True,
        verbose=True,
        **TrainableAligner.parse_parameters(mono_train_config_path)
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
        debug=True,
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
    print(TrainableAligner.parse_parameters(lda_train_config_path))
    a = TrainableAligner(
        corpus_directory=sick_corpus,
        dictionary_path=sick_dict,
        temporary_directory=data_directory,
        debug=True,
        verbose=True,
        **TrainableAligner.parse_parameters(lda_train_config_path)
    )
    a.train()


def test_sick_sat(sick_dict, sick_corpus, generated_dir, sat_train_config_path):
    data_directory = os.path.join(generated_dir, "temp", "sat_test")
    output_model_path = os.path.join(data_directory, "sat_model.zip")
    shutil.rmtree(data_directory, ignore_errors=True)
    a = TrainableAligner(
        **TrainableAligner.parse_parameters(sat_train_config_path),
        corpus_directory=sick_corpus,
        dictionary_path=sick_dict,
        debug=True,
        verbose=True,
        temporary_directory=data_directory
    )
    a.train()
    a.export_model(output_model_path)

    assert os.path.exists
