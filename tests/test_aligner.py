import os
import pytest
import shutil

from montreal_forced_aligner.aligner import TrainableAligner


#@pytest.mark.skip(reason='Optimization')
def test_sick_mono(sick_dict, sick_corpus, generated_dir, mono_train_config):
    shutil.rmtree(sick_corpus.output_directory, ignore_errors=True)
    os.makedirs(sick_corpus.output_directory, exist_ok=True)
    mono_train_config, align_config = mono_train_config
    print(mono_train_config.training_configs[0].feature_config.use_mp)
    data_directory = os.path.join(generated_dir, 'temp', 'mono_test')
    shutil.rmtree(data_directory, ignore_errors=True)
    a = TrainableAligner(sick_corpus, sick_dict, mono_train_config, align_config,
                         temp_directory=data_directory)
    a.train()


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


#@pytest.mark.skip(reason='Optimization')
def test_sick_ivector(sick_dict, sick_corpus, generated_dir, ivector_train_config):
    shutil.rmtree(sick_corpus.output_directory, ignore_errors=True)
    os.makedirs(sick_corpus.output_directory, exist_ok=True)
    ivector_train_config, align_config = ivector_train_config
    data_directory = os.path.join(generated_dir, 'temp', 'ivector_test')
    shutil.rmtree(data_directory, ignore_errors=True)
    a = TrainableAligner(sick_corpus, sick_dict, ivector_train_config, align_config,
                         temp_directory=data_directory)
    a.train()

