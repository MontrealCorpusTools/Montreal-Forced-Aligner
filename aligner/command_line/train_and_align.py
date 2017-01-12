import sys
from aligner.command_line.align import unfix_path
import shutil, os
import argparse
import multiprocessing as mp

from aligner.corpus import Corpus
from aligner.dictionary import Dictionary
from aligner.aligner import TrainableAligner
from aligner.utils import no_dictionary

TEMP_DIR = os.path.expanduser('~/Documents/MFA')


def align_corpus(corpus_dir, dict_path,  output_directory, temp_dir,
            output_model_path, args):
    if temp_dir == '':
        temp_dir = TEMP_DIR
    else:
        temp_dir = os.path.expanduser(temp_dir)
    corpus_name = os.path.basename(corpus_dir)
    if corpus_name == '':
        corpus_dir = os.path.dirname(corpus_dir)
        corpus_name = os.path.basename(corpus_dir)
    data_directory = os.path.join(temp_dir, corpus_name)
    if args.clean:
        shutil.rmtree(data_directory, ignore_errors = True)
        shutil.rmtree(output_directory, ignore_errors = True)

    os.makedirs(data_directory, exist_ok = True)
    os.makedirs(output_directory, exist_ok = True)

    corpus = Corpus(corpus_dir, data_directory, args.speaker_characters, num_jobs = args.num_jobs)
    print(corpus.speaker_utterance_info())
    corpus.write()
    corpus.create_mfccs()
    dictionary = Dictionary(dict_path, data_directory, word_set=corpus.word_set)
    dictionary.write()
    corpus.setup_splits(dictionary)
    utt_oov_path = os.path.join(corpus.split_directory, 'utterance_oovs.txt')
    if os.path.exists(utt_oov_path):
        shutil.copy(utt_oov_path, output_directory)
    oov_path = os.path.join(corpus.split_directory, 'oovs_found.txt')
    if os.path.exists(oov_path):
        shutil.copy(oov_path, output_directory)
    mono_params = {'align_often': not args.fast}
    tri_params = {'align_often': not args.fast}
    tri_fmllr_params = {'align_often': not args.fast}
    a = TrainableAligner(corpus, dictionary, output_directory,
                        temp_directory = data_directory,
                        mono_params = mono_params, tri_params = tri_params,
                        tri_fmllr_params = tri_fmllr_params, num_jobs = args.num_jobs)
    a.verbose = args.verbose
    a.train_mono()
    a.export_textgrids()
    a.train_tri()
    a.export_textgrids()
    a.train_tri_fmllr()
    a.export_textgrids()
    if output_model_path is not None:
        a.save(output_model_path)


def align_corpus_no_dict(corpus_dir, output_directory, temp_dir,
        output_model_path, args):
    if not temp_dir:
        temp_dir = TEMP_DIR
    else:
        temp_dir = os.path.expanduser(temp_dir)
    corpus_name = os.path.basename(corpus_dir)
    data_directory = os.path.join(temp_dir, corpus_name)
    if args.clean:
        shutil.rmtree(data_directory, ignore_errors = True)
        shutil.rmtree(output_directory, ignore_errors = True)

    os.makedirs(data_directory, exist_ok = True)
    os.makedirs(output_directory, exist_ok = True)

    corpus = Corpus(corpus_dir, data_directory, args.speaker_characters, num_jobs = args.num_jobs, debug=args.debug)
    print(corpus.speaker_utterance_info())
    dictionary = no_dictionary(corpus, data_directory)
    dictionary.write()
    corpus.write()
    corpus.create_mfccs()
    corpus.setup_splits(dictionary)
    mono_params = {'align_often': not args.fast}
    tri_params = {'align_often': not args.fast}
    tri_fmllr_params = {'align_often': not args.fast}
    a = TrainableAligner(corpus, dictionary, output_directory,
                        temp_directory = data_directory,
                        mono_params = mono_params, tri_params = tri_params,
                        tri_fmllr_params = tri_fmllr_params, num_jobs = args.num_jobs, debug=args.debug)
    a.verbose = args.verbose
    a.train_mono()
    a.export_textgrids()
    a.train_tri()
    a.export_textgrids()
    a.train_tri_fmllr()
    a.export_textgrids()
    if output_model_path is not None:
        a.save(output_model_path)


if __name__ == '__main__': # pragma: no cover
    mp.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_dir', help = 'Full path to the source directory to align')
    parser.add_argument('dict_path', help = 'Full path to the pronunciation dictionary to use', nargs='?', default = '')
    parser.add_argument('output_dir', help = 'Full path to output directory, will be created if it doesn\'t exist')
    parser.add_argument('-o', '--output_model_path', type = str, default = '', help = 'Full path to save resulting acoustic and dictionary model')
    parser.add_argument('-s', '--speaker_characters', type = int, default = 0,
                    help = 'Number of characters of filenames to use for determining speaker, default is to use directory names')
    parser.add_argument('-t', '--temp_directory', type = str, default = '',
                    help = 'Temporary directory root to use for aligning, default is ~/Documents/MFA')
    parser.add_argument('-f', '--fast', help = "Perform a quick alignment with half the number of alignment iterations", action = 'store_true')
    parser.add_argument('-j','--num_jobs', type = int, default = 3,
                    help = 'Number of cores to use while aligning')
    parser.add_argument('-v', '--verbose', help = "Output debug messages about alignment", action = 'store_true')
    parser.add_argument('--nodict', help = "Create a dictionary based on the orthography", action = 'store_true')
    parser.add_argument('-c', '--clean', help = "Remove files from previous runs", action = 'store_true')
    parser.add_argument('-d', '--debug', help = "Debug the aligner", action = 'store_true')
    args = parser.parse_args()
    corpus_dir = os.path.expanduser(args.corpus_dir)
    dict_path = os.path.expanduser(args.dict_path)
    output_dir = os.path.expanduser(args.output_dir)
    output_model_path = os.path.expanduser(args.output_model_path)
    temp_dir = args.temp_directory
    if not output_model_path:
        output_model_path = None
    if args.nodict == False and dict_path == '':
        raise(Exception('Must specify dictionary or nodict option'))
    if args.nodict == True and dict_path != '':
        raise(Exception('Dict_path cannot be specified with nodict option'))
    elif args.nodict == True:
        align_corpus_no_dict(corpus_dir, output_dir, temp_dir,
                    output_model_path, args)
    elif args.nodict == False:
        align_corpus(corpus_dir,dict_path, output_dir, temp_dir,
                    output_model_path, args)
    unfix_path()
