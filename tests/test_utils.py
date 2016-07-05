import os
import pytest

from aligner.corpus import Corpus
from aligner.config import MfccConfig
from aligner.dictionary import Dictionary
from aligner.utils import no_dictionary

def test_acoustic(basic_dict_path, basic_dir, generated_dir, output_directory):
    dictionary = Dictionary(basic_dict_path, os.path.join(generated_dir, 'acoustic'))
    dictionary.write()
    output_directory = os.path.join(generated_dir, 'acoustic')
    c = MfccConfig(output_directory)
    d = Corpus(basic_dir, output_directory, c)
    d.write()
    d.create_mfccs()
    d.setup_splits(dictionary)
    n = no_dictionary(d, output_directory)
    assert n.words['should'] == ['s h o u l d']
    assert '<vocnoise>' not in n.words
    assert n.words['here\'s'] == ['h e r e s']