import os
import pytest

from aligner.aligner import TrainableAligner


def test_sick_mono(sick_dict, sick_corpus, generated_dir, mono_train_config):
    mono_train_config, align_config = mono_train_config
    a = TrainableAligner(sick_corpus, sick_dict, mono_train_config, align_config,
                         os.path.join(generated_dir, 'sick_output'),
                         temp_directory=os.path.join(generated_dir, 'temp', 'mono_test'), skip_input=True)
    a.train()


def test_sick_tri(sick_dict, sick_corpus, generated_dir, tri_train_config):
    tri_train_config, align_config = tri_train_config
    a = TrainableAligner(sick_corpus, sick_dict, tri_train_config, align_config,
                         os.path.join(generated_dir, 'sick_output'),
                         temp_directory=os.path.join(generated_dir, 'temp', 'tri_test'), skip_input=True)
    a.train()


def test_sick_lda(sick_dict, sick_corpus, generated_dir, lda_train_config):
    lda_train_config, align_config = lda_train_config
    a = TrainableAligner(sick_corpus, sick_dict, lda_train_config, align_config,
                         os.path.join(generated_dir, 'sick_output'),
                         temp_directory=os.path.join(generated_dir, 'temp', 'lda_test'), skip_input=True)
    a.train()


def test_sick_sat(sick_dict, sick_corpus, generated_dir, sat_train_config):
    sat_train_config, align_config = sat_train_config
    a = TrainableAligner(sick_corpus, sick_dict, sat_train_config, align_config,
                         os.path.join(generated_dir, 'sick_output'),
                         temp_directory=os.path.join(generated_dir, 'temp', 'sat_test'), skip_input=True)
    a.train()
    a.export_textgrids()


def test_sick_lda_sat(sick_dict, sick_corpus, generated_dir, lda_sat_train_config):
    lda_sat_train_config, align_config = lda_sat_train_config
    a = TrainableAligner(sick_corpus, sick_dict, lda_sat_train_config, align_config,
                         os.path.join(generated_dir, 'sick_output'),
                         temp_directory=os.path.join(generated_dir, 'temp', 'lda_sat_test'), skip_input=True)
    a.train()
    a.export_textgrids()
