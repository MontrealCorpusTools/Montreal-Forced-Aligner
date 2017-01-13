import os

from aligner.corpus import Corpus
from aligner.utils import no_dictionary


def test_acoustic(basic_dir, generated_dir):
    output_directory = os.path.join(generated_dir, 'acoustic')
    d = Corpus(basic_dir, output_directory)
    d.write()
    d.create_mfccs()
    n = no_dictionary(d, output_directory)
    d.setup_splits(n)
    assert n.words['should'][0][0] == ('s', 'h', 'o', 'u', 'l', 'd')
    assert '<vocnoise>' not in n.words
    assert n.words['here\'s'][0][0] == ('h', 'e', 'r', 'e', 's')


def test_vietnamese(textgrid_directory, generated_dir):
    output_directory = os.path.join(generated_dir, 'vietnamese')
    d = Corpus(os.path.join(textgrid_directory, 'vietnamese'), output_directory)
    d.write()
    d.create_mfccs()
    n = no_dictionary(d, output_directory)
    d.setup_splits(n)
    assert n.words['chăn'][0][0] == ('c', 'h', 'ă', 'n')
    assert '<vocnoise>' not in n.words
    assert n.words['tập'][0][0] == ('t','ậ','p')
