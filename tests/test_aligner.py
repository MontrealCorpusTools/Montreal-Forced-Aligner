import os
import pytest

from aligner.aligner import TrainableAligner


def test_sick_mono(sick_dict, sick_corpus, generated_dir):
    a = TrainableAligner(sick_corpus, sick_dict, os.path.join(generated_dir, 'sick_output'),
                         temp_directory=os.path.join(generated_dir, 'sickcorpus'), skip_input=True)
    a.train_mono()
    a.export_textgrids()


def test_sick_tri(sick_dict, sick_corpus, generated_dir):
    a = TrainableAligner(sick_corpus, sick_dict, os.path.join(generated_dir, 'sick_output'),
                         temp_directory=os.path.join(generated_dir, 'sickcorpus'), skip_input=True)
    a.train_tri()
    a.export_textgrids()


def test_sick_tri_fmllr(sick_dict, sick_corpus, generated_dir):
    a = TrainableAligner(sick_corpus, sick_dict, os.path.join(generated_dir, 'sick_output'),
                         temp_directory=os.path.join(generated_dir, 'sickcorpus'), skip_input=True)
    a.train_tri_fmllr()
    a.export_textgrids()


def test_sick_lda_mllt(sick_dict, sick_corpus, generated_dir):
    a = TrainableAligner(sick_corpus, sick_dict, os.path.join(generated_dir, 'sick_output'),
                         temp_directory=os.path.join(generated_dir, 'sickcorpus'))
    a.train_lda_mllt()
    a.export_textgrids()

def test_sick_diag_ubm(sick_dict, sick_corpus, generated_dir):
    a = TrainableAligner(sick_corpus, sick_dict, os.path.join(generated_dir, 'sick_output'),
                         temp_directory=os.path.join(generated_dir, 'sickcorpus'))
    a.train_diag_ubm()

# Test to be integrated in future, when user training of the i-vector extractor is supported.
# Currently, the test corpus is too small.
"""def test_sick_ivectors(sick_dict, sick_corpus, generated_dir):
    a = TrainableAligner(sick_corpus, sick_dict, os.path.join(generated_dir, 'sick_output'),
                         temp_directory=os.path.join(generated_dir, 'sickcorpus'))
    a.ivector_extractor()"""

def test_sick_nnet_basic(sick_dict, sick_corpus, generated_dir):
    a = TrainableAligner(sick_corpus, sick_dict, os.path.join(generated_dir, 'sick_output'),
                         temp_directory=os.path.join(generated_dir, 'sickcorpus'))
    a.train_nnet_basic()
    a.export_textgrids()
