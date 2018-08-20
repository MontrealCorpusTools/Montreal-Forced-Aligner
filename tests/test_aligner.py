import os
import pytest
import shutil

from aligner.aligner import TrainableAligner


def test_sick_nnet(sick_dict, sick_corpus, generated_dir, nnet_train_config):
    shutil.rmtree(sick_corpus.output_directory, ignore_errors=True)
    os.makedirs(sick_corpus.output_directory, exist_ok=True)
    nnet_train_config, align_config = nnet_train_config
    data_directory = os.path.join(generated_dir, 'temp', 'nnet_test')
    shutil.rmtree(data_directory, ignore_errors=True)
    a = TrainableAligner(sick_corpus, sick_dict, nnet_train_config, align_config,
                         os.path.join(generated_dir, 'sick_output'),
                         temp_directory=data_directory)
    a.train()


def test_sick_nnet_ivectors(sick_dict, sick_corpus, generated_dir, nnet_ivectors_train_config):
    shutil.rmtree(sick_corpus.output_directory, ignore_errors=True)
    os.makedirs(sick_corpus.output_directory, exist_ok=True)
    nnet_train_config, align_config = nnet_ivectors_train_config
    data_directory = os.path.join(generated_dir, 'temp', 'nnet_test')
    shutil.rmtree(data_directory, ignore_errors=True)
    a = TrainableAligner(sick_corpus, sick_dict, nnet_train_config, align_config,
                         os.path.join(generated_dir, 'sick_output'),
                         temp_directory=data_directory)
    a.train()


def test_sick_ivector(sick_dict, sick_corpus, generated_dir, ivector_train_config):
    shutil.rmtree(sick_corpus.output_directory, ignore_errors=True)
    os.makedirs(sick_corpus.output_directory, exist_ok=True)
    ivector_train_config, align_config = ivector_train_config
    data_directory = os.path.join(generated_dir, 'temp', 'ivector_test')
    shutil.rmtree(data_directory, ignore_errors=True)
    a = TrainableAligner(sick_corpus, sick_dict, ivector_train_config, align_config,
                         os.path.join(generated_dir, 'sick_output'),
                         temp_directory=data_directory)
    a.train()


def test_sick_mono(sick_dict, sick_corpus, generated_dir, mono_train_config):
    shutil.rmtree(sick_corpus.output_directory, ignore_errors=True)
    os.makedirs(sick_corpus.output_directory, exist_ok=True)
    mono_train_config, align_config = mono_train_config
    data_directory = os.path.join(generated_dir, 'temp', 'mono_test')
    shutil.rmtree(data_directory, ignore_errors=True)
    a = TrainableAligner(sick_corpus, sick_dict, mono_train_config, align_config,
                         os.path.join(generated_dir, 'sick_output'),
                         temp_directory=data_directory)
    a.train()


def test_sick_tri(sick_dict, sick_corpus, generated_dir, tri_train_config):
    shutil.rmtree(sick_corpus.output_directory, ignore_errors=True)
    os.makedirs(sick_corpus.output_directory, exist_ok=True)
    tri_train_config, align_config = tri_train_config
    data_directory = os.path.join(generated_dir, 'temp', 'tri_test')
    shutil.rmtree(data_directory, ignore_errors=True)
    a = TrainableAligner(sick_corpus, sick_dict, tri_train_config, align_config,
                         os.path.join(generated_dir, 'sick_output'),
                         temp_directory=data_directory)
    a.train()


def test_sick_lda(sick_dict, sick_corpus, generated_dir, lda_train_config):
    shutil.rmtree(sick_corpus.output_directory, ignore_errors=True)
    os.makedirs(sick_corpus.output_directory, exist_ok=True)
    lda_train_config, align_config = lda_train_config
    data_directory = os.path.join(generated_dir, 'temp', 'lda_test')
    shutil.rmtree(data_directory, ignore_errors=True)
    a = TrainableAligner(sick_corpus, sick_dict, lda_train_config, align_config,
                         os.path.join(generated_dir, 'sick_output'),
                         temp_directory=data_directory)
    a.train()


def test_sick_sat(sick_dict, sick_corpus, generated_dir, sat_train_config):
    shutil.rmtree(sick_corpus.output_directory, ignore_errors=True)
    os.makedirs(sick_corpus.output_directory, exist_ok=True)
    sat_train_config, align_config = sat_train_config
    data_directory = os.path.join(generated_dir, 'temp', 'sat_test')
    shutil.rmtree(data_directory, ignore_errors=True)
    a = TrainableAligner(sick_corpus, sick_dict, sat_train_config, align_config,
                         os.path.join(generated_dir, 'sick_output'),
                         temp_directory=data_directory)
    a.train()
    a.export_textgrids()


def test_sick_lda_sat(sick_dict, sick_corpus, generated_dir, lda_sat_train_config):
    shutil.rmtree(sick_corpus.output_directory, ignore_errors=True)
    os.makedirs(sick_corpus.output_directory, exist_ok=True)
    lda_sat_train_config, align_config = lda_sat_train_config
    data_directory = os.path.join(generated_dir, 'temp', 'lda_sat_test')
    shutil.rmtree(data_directory, ignore_errors=True)
    a = TrainableAligner(sick_corpus, sick_dict, lda_sat_train_config, align_config,
                         os.path.join(generated_dir, 'sick_output'),
                         temp_directory=data_directory)
    a.train()
    a.export_textgrids()