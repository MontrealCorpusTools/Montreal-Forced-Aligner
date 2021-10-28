import os
import pytest
import shutil

from montreal_forced_aligner.aligner import TrainableAligner, PretrainedAligner
from montreal_forced_aligner.models import AcousticModel


#@pytest.mark.skip(reason='Optimization')
def test_sick_mono(sick_dict, sick_corpus, generated_dir, mono_train_config, mono_align_model_path,
                   mono_align_config, mono_output_directory):
    shutil.rmtree(sick_corpus.output_directory, ignore_errors=True)
    os.makedirs(sick_corpus.output_directory, exist_ok=True)
    mono_train_config, align_config = mono_train_config
    print(mono_train_config.training_configs[0].feature_config.use_mp)
    data_directory = os.path.join(generated_dir, 'temp', 'mono_train_test')
    shutil.rmtree(data_directory, ignore_errors=True)
    a = TrainableAligner(sick_corpus, sick_dict, mono_train_config, align_config,
                         temp_directory=data_directory)
    a.train()
    a.save(mono_align_model_path)

    model = AcousticModel(mono_align_model_path)
    data_directory = os.path.join(generated_dir, 'temp', 'mono_align_test')
    shutil.rmtree(data_directory, ignore_errors=True)
    mono_align_config.debug = True
    a = PretrainedAligner(sick_corpus, sick_dict, model, mono_align_config,
                          temp_directory=data_directory, debug=True)
    a.align()
    a.export_textgrids(mono_output_directory)


#@pytest.mark.skip(reason='Optimization')
def test_sick_tri(sick_dict, sick_corpus, generated_dir, tri_train_config):
    shutil.rmtree(sick_corpus.output_directory, ignore_errors=True)
    os.makedirs(sick_corpus.output_directory, exist_ok=True)
    tri_train_config, align_config = tri_train_config
    data_directory = os.path.join(generated_dir, 'temp', 'tri_test')
    shutil.rmtree(data_directory, ignore_errors=True)
    a = TrainableAligner(sick_corpus, sick_dict, tri_train_config, align_config,
                         temp_directory=data_directory)
    a.train()


def test_sick_lda(sick_dict, sick_corpus, generated_dir, lda_train_config):
    shutil.rmtree(sick_corpus.output_directory, ignore_errors=True)
    os.makedirs(sick_corpus.output_directory, exist_ok=True)
    lda_train_config, align_config = lda_train_config
    data_directory = os.path.join(generated_dir, 'temp', 'lda_test')
    shutil.rmtree(data_directory, ignore_errors=True)
    a = TrainableAligner(sick_corpus, sick_dict, lda_train_config, align_config,
                         temp_directory=data_directory)
    a.train()


def test_sick_sat(sick_dict, sick_corpus, generated_dir, sat_train_config):
    shutil.rmtree(sick_corpus.output_directory, ignore_errors=True)
    os.makedirs(sick_corpus.output_directory, exist_ok=True)
    sat_train_config, align_config = sat_train_config
    data_directory = os.path.join(generated_dir, 'temp', 'sat_test')
    shutil.rmtree(data_directory, ignore_errors=True)
    a = TrainableAligner(sick_corpus, sick_dict, sat_train_config, align_config,
                         temp_directory=data_directory)
    a.train()
    a.export_textgrids(os.path.join(generated_dir, 'sick_output'))

