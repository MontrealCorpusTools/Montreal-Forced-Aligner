from aligner.command_line.align import fix_path, unfix_path, PRETRAINED_LANGUAGES
import shutil
import os
import argparse
import multiprocessing as mp

from aligner.corpus import Corpus
from aligner.dictionary import Dictionary
from aligner.validator import CorpusValidator
from aligner.exceptions import ArgumentError
from aligner.config import TEMP_DIR


def validate_corpus(args):
    if not args.temp_directory:
        temp_dir = TEMP_DIR
    else:
        temp_dir = os.path.expanduser(args.temp_directory)
    corpus_name = os.path.basename(args.corpus_directory)
    if corpus_name == '':
        args.corpus_directory = os.path.dirname(args.corpus_directory)
        corpus_name = os.path.basename(args.corpus_directory)
    data_directory = os.path.join(temp_dir, corpus_name)
    shutil.rmtree(data_directory, ignore_errors=True)

    os.makedirs(data_directory, exist_ok=True)

    corpus = Corpus(args.corpus_directory, data_directory, speaker_characters=args.speaker_characters,
                    num_jobs=getattr(args, 'num_jobs', 3))
    dictionary = Dictionary(args.dictionary_path, data_directory, word_set=corpus.word_set)

    a = CorpusValidator(corpus, dictionary, temp_directory=data_directory,
                        ignore_acoustics=getattr(args, 'ignore_acoustics', False),
                        test_transcriptions=getattr(args, 'test_transcriptions', False))
    a.validate()


def validate_args(args):
    if args.test_transcriptions and args.ignore_acoustics:
        raise ArgumentError('Cannot test transcriptions without acoustic feature generation.')
    if not os.path.exists(args.corpus_directory):
        raise (ArgumentError('Could not find the corpus directory {}.'.format(args.corpus_directory)))
    if not os.path.isdir(args.corpus_directory):
        raise (ArgumentError('The specified corpus directory ({}) is not a directory.'.format(args.corpus_directory)))
    if not os.path.exists(args.dictionary_path):
        raise (ArgumentError('Could not find the dictionary file {}'.format(args.dictionary_path)))
    if not os.path.isfile(args.dictionary_path):
        raise (ArgumentError('The specified dictionary path ({}) is not a text file.'.format(args.dictionary_path)))


if __name__ == '__main__':  # pragma: no cover
    mp.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_directory', help='Full path to the source directory to align')
    parser.add_argument('dictionary_path', help='Full path to the pronunciation dictionary to use',
                        default='')
    parser.add_argument('acoustic_model_path', help='Full path to the archive containing pre-trained model or language ({})'.format(
                            ', '.join(PRETRAINED_LANGUAGES)), nargs='?',
                        default='')
    parser.add_argument('-s', '--speaker_characters', type=str, default='0',
                        help='Number of characters of file names to use for determining speaker, '
                             'default is to use directory names')
    parser.add_argument('-t', '--temp_directory', type=str, default='',
                        help='Temporary directory root to use for aligning, default is ~/Documents/MFA')
    parser.add_argument('--test_transcriptions', help="Test accuracy of transcriptions", action='store_true')
    parser.add_argument('--ignore_acoustics', help="Skip acoustic feature generation and associated validation", action='store_true')
    parser.add_argument('-j', '--num_jobs', type=int, default=3,
                        help='Number of cores to use while aligning')

    args = parser.parse_args()
    fix_path()
    try:
        args.speaker_characters = int(args.speaker_characters)
    except ValueError:
        pass
    validate_args(args)
    temp_dir = args.temp_directory
    validate_corpus(args)
    unfix_path()
