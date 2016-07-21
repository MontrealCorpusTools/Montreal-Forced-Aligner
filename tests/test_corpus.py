import os
import pytest

from aligner.corpus import Corpus
from aligner.config import MfccConfig
from aligner.dictionary import Dictionary

def test_basic(basic_dict_path, basic_dir, generated_dir):
    dictionary = Dictionary(basic_dict_path, os.path.join(generated_dir, 'basic'))
    dictionary.write()
    output_directory = os.path.join(generated_dir, 'basic')
    d = Corpus(basic_dir, output_directory)
    d.write()
    d.create_mfccs()
    d.setup_splits(dictionary)
    assert(d.get_feat_dim() == '39')
