from aligner.command_line.align import fix_path, unfix_path
import shutil
import os
import argparse
import multiprocessing as mp
import yaml
import time

from aligner import __version__
from aligner.corpus import Corpus
from aligner.dictionary import Dictionary
from aligner.aligner import TrainableAligner
from aligner.utils import no_dictionary
from aligner.config import TEMP_DIR


def align_corpus(args, skip_input=False):
    if not args.temp_directory:
        temp_dir = TEMP_DIR
    else:
        temp_dir = os.path.expanduser(args.temp_directory)
    corpus_name = os.path.basename(args.corpus_directory)
    if corpus_name == '':
        args.corpus_directory = os.path.dirname(args.corpus_directory)
        corpus_name = os.path.basename(args.corpus_directory)
    data_directory = os.path.join(temp_dir, corpus_name)
    conf_path = os.path.join(data_directory, 'config.yml')
    if os.path.exists(conf_path):
        with open(conf_path, 'r') as f:
            conf = yaml.load(f)
    else:
        conf = {'dirty': False,
                'begin': time.time(),
                'version': __version__,
                'type': 'train_and_align',
                'corpus_directory': args.corpus_directory,
                'dictionary_path': args.dictionary_path}
    if getattr(args, 'clean', False) \
            or conf['dirty'] or conf['type'] != 'train_and_align' \
            or conf['corpus_directory'] != args.corpus_directory \
            or conf['version'] != __version__ \
            or conf['dictionary_path'] != args.dictionary_path:
        shutil.rmtree(data_directory, ignore_errors=True)
        shutil.rmtree(args.output_directory, ignore_errors=True)

    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(args.output_directory, exist_ok=True)
    try:
        corpus = Corpus(args.corpus_directory, data_directory, speaker_characters=args.speaker_characters,
                        num_jobs=getattr(args, 'num_jobs', 3),
                        debug=getattr(args, 'debug', False),
                        ignore_exceptions=getattr(args, 'ignore_exceptions', False))
        dictionary = Dictionary(args.dictionary_path, data_directory, word_set=corpus.word_set)
        utt_oov_path = os.path.join(corpus.split_directory, 'utterance_oovs.txt')
        if os.path.exists(utt_oov_path):
            shutil.copy(utt_oov_path, args.output_directory)
        oov_path = os.path.join(corpus.split_directory, 'oovs_found.txt')
        if os.path.exists(oov_path):
            shutil.copy(oov_path, args.output_directory)
        mono_params = {'align_often': not args.fast}
        tri_params = {'align_often': not args.fast}
        tri_fmllr_params = {'align_often': not args.fast}
        a = TrainableAligner(corpus, dictionary, args.output_directory,
                             temp_directory=data_directory,
                             mono_params=mono_params, tri_params=tri_params,
                             tri_fmllr_params=tri_fmllr_params, num_jobs=args.num_jobs)
        a.verbose = args.verbose

        # GMM training (looks like it needs to be done either way, as a starter for nnet)
        a.train_mono()
        a.export_textgrids()
        a.train_tri()
        a.export_textgrids()
        a.train_tri_fmllr()
        a.export_textgrids()

        # nnet training
        if getattr(args, 'neural_net', True):
            # Do nnet training
            a.train_lda_mllt()  # Implemented!
            a.train_diag_ubm()  # Implemented!
            a.ivectors()        # NOT YET IMPLEMENTED
            a.train_nnet_basic()      # NOT YET IMPLEMENTED
            a.export_textgrids()

        if args.output_model_path is not None:
            a.save(args.output_model_path)
    except:
        conf['dirty'] = True
        raise
    finally:
        with open(conf_path, 'w') as f:
            yaml.dump(conf, f)


def align_corpus_no_dict(args, skip_input=False):
    if not args.temp_directory:
        temp_dir = TEMP_DIR
    else:
        temp_dir = os.path.expanduser(args.temp_directory)
    corpus_name = os.path.basename(args.corpus_directory)
    data_directory = os.path.join(temp_dir, corpus_name)
    if args.clean:
        shutil.rmtree(data_directory, ignore_errors=True)
        shutil.rmtree(args.output_directory, ignore_errors=True)

    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(args.output_directory, exist_ok=True)

    corpus = Corpus(args.corpus_directory, data_directory, args.speaker_characters,
                    num_jobs=getattr(args, 'num_jobs', 3),
                    debug=getattr(args, 'debug', False),
                    ignore_exceptions=getattr(args, 'ignore_exceptions', False))
    print(corpus.speaker_utterance_info())
    dictionary = no_dictionary(corpus, data_directory)
    mono_params = {'align_often': not args.fast}
    tri_params = {'align_often': not args.fast}
    tri_fmllr_params = {'align_often': not args.fast}
    a = TrainableAligner(corpus, dictionary, args.output_directory,
                         temp_directory=data_directory,
                         mono_params=mono_params, tri_params=tri_params,
                         tri_fmllr_params=tri_fmllr_params, num_jobs=args.num_jobs, debug=args.debug)
    a.verbose = args.verbose
    a.train_mono()
    a.export_textgrids()
    a.train_tri()
    a.export_textgrids()
    a.train_tri_fmllr()
    a.export_textgrids()
    if args.output_model_path is not None:
        a.save(args.output_model_path)


def validate_args(args):
    if not args.no_dict and args.dictionary_path == '':
        raise (Exception('Must specify dictionary or no_dict option'))
    if args.no_dict and args.dictionary_path != '':
        raise (Exception('Dict_path cannot be specified with no_dict option'))


if __name__ == '__main__':  # pragma: no cover
    mp.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_directory', help='Full path to the source directory to align')
    parser.add_argument('dictionary_path', help='Full path to the pronunciation dictionary to use', nargs='?', default='')
    parser.add_argument('output_directory', help="Full path to output directory, will be created if it doesn't exist")

    parser.add_argument('-n', '--neural_net', action='store_true')

    parser.add_argument('-o', '--output_model_path', type=str, default='',
                        help='Full path to save resulting acoustic and dictionary model')
    parser.add_argument('-s', '--speaker_characters', type=str, default='0',
                        help='Number of characters of filenames to use for determining speaker, '
                             'default is to use directory names')
    parser.add_argument('-t', '--temp_directory', type=str, default='',
                        help='Temporary directory root to use for aligning, default is ~/Documents/MFA')
    parser.add_argument('-f', '--fast', help="Perform a quick alignment with half the number of alignment iterations",
                        action='store_true')
    parser.add_argument('-j', '--num_jobs', type=int, default=3,
                        help='Number of cores to use while aligning')
    parser.add_argument('-v', '--verbose', help="Output debug messages about alignment", action='store_true')
    parser.add_argument('--no_dict', help="Create a dictionary based on the orthography", action='store_true')
    parser.add_argument('-c', '--clean', help="Remove files from previous runs", action='store_true')
    parser.add_argument('-d', '--debug', help="Debug the aligner", action='store_true')
    parser.add_argument('-i', '--ignore_exceptions', help='Ignore exceptions raised when parsing data',
                        action='store_true')
    args = parser.parse_args()
    fix_path()
    try:
        args.speaker_characters = int(args.speaker_characters)
    except ValueError:
        pass
    validate_args(args)
    if not args.output_model_path:
        args.output_model_path = None
    temp_dir = args.temp_directory
    if args.no_dict:
        align_corpus_no_dict(args)
    else:
        align_corpus(args)
    unfix_path()
