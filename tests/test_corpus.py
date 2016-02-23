import os
import pytest

from aligner.corpus import Corpus
from aligner.config import MfccConfig

def test_basic(basic_dir, generated_dir):
    output_directory = os.path.join(generated_dir, 'basic')
    c = MfccConfig(output_directory)
    d = Corpus(basic_dir, output_directory, c)
    d.write()
    d.create_mfccs()
