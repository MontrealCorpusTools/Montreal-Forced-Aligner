import sys
import shutil, os
import argparse

from aligner.corpus import Corpus
from aligner.config import MfccConfig
from aligner.dictionary import Dictionary
from aligner.aligner import PretrainedAligner
from aligner.archive import Archive

TEMP_DIR = os.path.expanduser('~/Documents/MFA')

def align_corpus(model_path, corpus_dir,  output_directory, speaker_characters, num_jobs, verbose):
    corpus_name = os.path.basename(corpus_dir)
    c = MfccConfig(os.path.join(TEMP_DIR, corpus_name))
    corpus = Corpus(corpus_dir, os.path.join(TEMP_DIR, corpus_name), c, speaker_characters, num_jobs = num_jobs)
    corpus.write()
    corpus.create_mfccs()
    archive = Archive(model_path)
    a = PretrainedAligner(archive, corpus, output_directory,
                        temp_directory = os.path.join(TEMP_DIR, corpus_name), num_jobs = num_jobs)
    a.verbose = verbose
    corpus.setup_splits(a.dictionary)
    a.do_align()
    a.export_textgrids()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help = 'Full path to the archive containing pre-trained model')
    parser.add_argument('corpus_dir', help = 'Full path to the directory to align')
    parser.add_argument('output_dir', help = 'Full path to output directory, will be created if it doesn\'t exist')
    parser.add_argument('-s', '--speaker_characters', type = int, default = 0,
                    help = 'Number of characters of filenames to use for determining speaker, default is to use directory names')
    parser.add_argument('-j','--num_jobs', type = int, default = 3,
                    help = 'Number of cores to use while aligning')
    parser.add_argument('-v', '--verbose', help = "Output debug messages about alignment", action = 'store_true')
    args = parser.parse_args()
    corpus_dir = args.corpus_dir
    model_path = args.model_path
    output_dir = args.output_dir
    align_corpus(model_path, corpus_dir, output_dir,
        args.speaker_characters, args.num_jobs, args.verbose)

