import os
import shutil

from montreal_forced_aligner.acoustic_modeling.trainer import TrainableAligner
from montreal_forced_aligner.alignment import PretrainedAligner


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
):
    a = TrainableAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=mixed_dict_path,
        **TrainableAligner.parse_parameters(mono_train_config_path)
    )
    a.train()
    a.export_model(mono_align_model_path)
    assert os.path.exists(mono_align_model_path)

    a = PretrainedAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=mixed_dict_path,
        acoustic_model_path=mono_align_model_path,
        **PretrainedAligner.parse_parameters(mono_align_config_path)
    )
    a.align()
    a.export_files(mono_output_directory)
    assert os.path.exists(mono_output_directory)


def test_pronunciation_training(
    mixed_dict_path, basic_corpus_dir, generated_dir, pron_train_config_path
):
    export_path = os.path.join(generated_dir, "pron_train_test_export", "model.zip")
    a = TrainableAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=mixed_dict_path,
        **TrainableAligner.parse_parameters(pron_train_config_path)
    )
    a.train()

    a.cleanup()
    assert not os.path.exists(export_path)
    assert not os.path.exists(
        os.path.join(generated_dir, "pron_train_test_export", os.path.basename(mixed_dict_path))
    )
    a = TrainableAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=mixed_dict_path,
        **TrainableAligner.parse_parameters(pron_train_config_path)
    )
    a.train()
    a.export_model(export_path)
    assert os.path.exists(export_path)
    assert os.path.exists(
        os.path.join(
            generated_dir,
            "pron_train_test_export",
            os.path.basename(mixed_dict_path).replace(".txt", ".dict"),
        )
    )


def test_pitch_feature_training(basic_dict_path, basic_corpus_dir, pitch_train_config_path):
    a = TrainableAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=basic_dict_path,
        debug=True,
        verbose=True,
        **TrainableAligner.parse_parameters(pitch_train_config_path)
    )
    assert a.use_pitch
    a.train()
    assert a.get_feat_dim() == 48


def test_basic_lda(basic_dict_path, basic_corpus_dir, lda_train_config_path):
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


def test_basic_sat(basic_dict_path, basic_corpus_dir, generated_dir, sat_train_config_path):
    data_directory = os.path.join(generated_dir, "sat_test")
    output_model_path = os.path.join(data_directory, "sat_model.zip")
    shutil.rmtree(data_directory, ignore_errors=True)
    a = TrainableAligner(
        **TrainableAligner.parse_parameters(sat_train_config_path),
        corpus_directory=basic_corpus_dir,
        dictionary_path=basic_dict_path,
        disable_mp=False,
        temporary_directory=data_directory
    )
    a.train()
    assert len(a.training_configs[a.final_identifier].fmllr_iterations) > 1
    a.export_model(output_model_path)

    assert os.path.exists(output_model_path)
    assert os.path.exists(os.path.join(a.output_directory, "sat", "trans.1.0.ark"))
