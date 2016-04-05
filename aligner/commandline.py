import sys
import shutil, os
import argparse

from aligner.corpus import Corpus
from aligner.config import MfccConfig
from aligner.dictionary import Dictionary
from aligner.aligner import BaseAligner

TEMP_DIR = os.path.expanduser('~/Documents/MFA')

def align_corpus(corpus_dir, dict_path,  output_directory, speaker_characters):
    corpus_name = os.path.basename(corpus_dir)
    dictionary = Dictionary(dict_path, os.path.join(TEMP_DIR, corpus_name))
    dictionary.write()
    c = MfccConfig(TEMP_DIR)
    corpus = Corpus(corpus_dir, os.path.join(TEMP_DIR, corpus_name), c, speaker_characters)
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
    parser.add_argument('corpus_dir', help = 'Full path to the source directory to align')
    parser.add_argument('dict_path', help = 'Full path to the pronunciation dictionary to use')
    parser.add_argument('output_dir', help = 'Full path to output directory, will be created if it doesn\'t exist')
    parser.add_argument('-s', '--speaker_characters', type = int, default = 0, 
                    help = 'Number of characters of filenames to use for determining speaker, default is to use directory names')
    args = parser.parse_args()
    corpus_dir = args.corpus_dir
    dict_path = args.dict_path
    output_dir = args.output_dir
    align_corpus(corpus_dir,dict_path, output_dir, args.speaker_characters)

