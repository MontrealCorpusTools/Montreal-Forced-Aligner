import sys
import shutil, os
import argparse

base = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,base)
from aligner.corpus import Corpus
from aligner.config import MfccConfig
from aligner.dictionary import Dictionary
from aligner.aligner import BaseAligner

TEMP_DIR = os.path.expanduser('~/Documents/MFA')

def align_corpus(corpus_dir, dict_path,  output_directory):
    corpus_name = os.path.basename(corpus_dir)
    dictionary = Dictionary(dict_path, os.path.join(TEMP_DIR, corpus_name))
    dictionary.write()
    c = MfccConfig(TEMP_DIR)
    corpus = Corpus(corpus_dir, os.path.join(TEMP_DIR, corpus_name), c)
    corpus.write()
    corpus.create_mfccs()
    corpus.setup_splits(dictionary)
    a = BaseAligner(corpus, dictionary, output_directory,
                        temp_directory = os.path.join(TEMP_DIR, corpus_name))
    a.train_mono()
    a.train_tri()
    a.train_tri_fmllr()
    a.export_textgrids()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_dir')
    parser.add_argument('dict_path')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    corpus_dir = args.corpus_dir
    dict_path = args.dict_path
    output_dir = args.output_dir
    align_corpus(corpus_dir,dict_path, output_dir)

