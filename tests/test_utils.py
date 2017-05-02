import os

from aligner.corpus import Corpus
from aligner.utils import no_dictionary


def test_acoustic(basic_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, 'acoustic')
    d = Corpus(basic_corpus_dir, output_directory)
    n = no_dictionary(d, output_directory)
    d.initialize_corpus(n)
    assert n.words['should'][0][0] == ('s', 'h', 'o', 'u', 'l', 'd')
    assert '<vocnoise>' not in n.words
    assert n.words['here\'s'][0][0] == ('h', 'e', 'r', 'e', 's')


def test_vietnamese(vietnamese_corpus_dir, temp_dir):
    output_directory = os.path.join(temp_dir, 'vietnamese')
    d = Corpus(vietnamese_corpus_dir, output_directory)
    n = no_dictionary(d, output_directory)
    d.initialize_corpus(n)
    assert n.words['chăn'][0][0] == ('c', 'h', 'ă', 'n')
    assert '<vocnoise>' not in n.words
    assert n.words['tập'][0][0] == ('t','ậ','p')
