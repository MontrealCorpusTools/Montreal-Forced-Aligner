import sys
import shutil, os
import argparse

from aligner.corpus import Corpus
from aligner.config import MfccConfig
from aligner.dictionary import Dictionary
from aligner.aligner import TrainableAligner
from aligner.utils import no_dictionary

TEMP_DIR = os.path.expanduser('~/Documents/MFA')

def align_corpus(corpus_dir, dict_path,  output_directory, speaker_characters, fast,
            output_model_path, num_jobs, verbose):
    corpus_name = os.path.basename(corpus_dir)
    dictionary = Dictionary(dict_path, os.path.join(TEMP_DIR, corpus_name))
    dictionary.write()
    c = MfccConfig(os.path.join(TEMP_DIR, corpus_name))
    corpus = Corpus(corpus_dir, os.path.join(TEMP_DIR, corpus_name), c, speaker_characters, num_jobs = num_jobs)
    corpus.write()
    corpus.create_mfccs()
    corpus.setup_splits(dictionary)
    mono_params = {'align_often': not fast}
    tri_params = {'align_often': not fast}
    tri_fmllr_params = {'align_often': not fast}
    a = TrainableAligner(corpus, dictionary, output_directory,
                        temp_directory = os.path.join(TEMP_DIR, corpus_name),
                        mono_params = mono_params, tri_params = tri_params,
                        tri_fmllr_params = tri_fmllr_params, num_jobs = num_jobs)
    a.verbose = verbose
    a.train_mono()
    a.export_textgrids()
    a.train_tri()
    a.export_textgrids()
    a.train_tri_fmllr()
    a.export_textgrids()
    if output_model_path is not None:
        a.save(output_model_path)

def align_corpus_no_dict(corpus_dir, output_directory, speaker_characters, fast,
        output_model_path, num_jobs, verbose):
    corpus_name = os.path.basename(corpus_dir)
    c = MfccConfig(os.path.join(TEMP_DIR, corpus_name))
    corpus = Corpus(corpus_dir, os.path.join(TEMP_DIR, corpus_name), c, speaker_characters, num_jobs = num_jobs)
    dictionary = no_dictionary(corpus, os.path.join(TEMP_DIR, corpus_name))
    corpus.write()
    corpus.create_mfccs()
    corpus.setup_splits(dictionary)
    mono_params = {'align_often': not fast}
    tri_params = {'align_often': not fast}
    tri_fmllr_params = {'align_often': not fast}
    a = TrainableAligner(corpus, dictionary, output_directory,
                        temp_directory = os.path.join(TEMP_DIR, corpus_name),
                        mono_params = mono_params, tri_params = tri_params,
                        tri_fmllr_params = tri_fmllr_params, num_jobs = num_jobs)
    a.verbose = verbose
    a.train_mono()
    a.export_textgrids()
    a.train_tri()
    a.export_textgrids()
    a.train_tri_fmllr()
    a.export_textgrids()
    if output_model_path is not None:
        a.save(output_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_dir', help = 'Full path to the source directory to align')
    parser.add_argument('dict_path', help = 'Full path to the pronunciation dictionary to use')
    parser.add_argument('output_dir', help = 'Full path to output directory, will be created if it doesn\'t exist')
    parser.add_argument('-o', '--output_model_path', type = str, default = '', help = 'Full path to save resulting acoustic and dictionary model')
    parser.add_argument('-s', '--speaker_characters', type = int, default = 0,
                    help = 'Number of characters of filenames to use for determining speaker, default is to use directory names')
    parser.add_argument('-f', '--fast', help = "Perform a quick alignment with half the number of alignment iterations", action = 'store_true')
    parser.add_argument('-j','--num_jobs', type = int, default = 3,
                    help = 'Number of cores to use while aligning')
    parser.add_argument('-v', '--verbose', help = "Output debug messages about alignment", action = 'store_true')
    parser.add_argument('--nodict', help = "Create a dictionary based on the orthography", action = 'store_true')
    args = parser.parse_args()
    corpus_dir = args.corpus_dir
    dict_path = args.dict_path
    output_dir = args.output_dir
    output_model_path = args.output_model_path
    if not output_model_path:
        output_model_path = None
    if args.nodict == True:
        align_corpus_no_dict(corpus_dir, output_dir, args.speaker_characters,
                    args.fast,
                    output_model_path, args.num_jobs, args.verbose)
    if args.nodict == False:
        align_corpus(corpus_dir,dict_path, output_dir, args.speaker_characters,
                    args.fast,
                    output_model_path, args.num_jobs, args.verbose)

