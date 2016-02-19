import os
import pytest

from aligner.dictionary import Dictionary

def test_basic(basic_dict_path, generated_dir):
    d = Dictionary(basic_dict_path, os.path.join(generated_dir, 'basic'))
    d.write()
