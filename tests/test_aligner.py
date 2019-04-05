import os
import pytest

from aligner.aligner import TrainableAligner


def test_sick_mono(sick_dict, sick_corpus, generated_dir):
    a = TrainableAligner(sick_corpus, sick_dict, os.path.join(generated_dir, 'sick_output'),
                         temp_directory=os.path.join(generated_dir, 'sickcorpus'), skip_input=True)
    a.train_mono()


def test_sick_tri(sick_dict, sick_corpus, generated_dir):
    a = TrainableAligner(sick_corpus, sick_dict, os.path.join(generated_dir, 'sick_output'),
                         temp_directory=os.path.join(generated_dir, 'sickcorpus'), skip_input=True)
    a.train_tri()


def test_sick_tri_fmllr(sick_dict, sick_corpus, generated_dir):
    a = TrainableAligner(sick_corpus, sick_dict, os.path.join(generated_dir, 'sick_output'),
                         temp_directory=os.path.join(generated_dir, 'sickcorpus'), skip_input=True)
    a.train_tri_fmllr()
    a.export_textgrids()
