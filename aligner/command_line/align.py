import sys
import shutil
import os
import time
import argparse
import multiprocessing as mp
import yaml

from aligner import __version__
from aligner.corpus import Corpus
from aligner.dictionary import Dictionary
from aligner.aligner import PretrainedAligner
from aligner.models import AcousticModel
from aligner.config import TEMP_DIR, align_yaml_to_config, load_basic_align

from aligner.exceptions import ArgumentError


class DummyArgs(object):
    def __init__(self):
        self.speaker_characters = 0
        self.num_jobs = 0
        self.verbose = False
        self.clean = True
        self.fast = True
        self.no_speaker_adaptation = False
        self.debug = False
        self.errors = False
        self.temp_directory = None


def fix_path():
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
        if sys.platform == 'win32':
            thirdparty_dir = os.path.join(base_dir, 'thirdparty', 'bin')
        else:
            thirdparty_dir = os.path.join(base_dir, 'thirdparty', 'bin')
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        thirdparty_dir = os.path.join(base_dir, 'thirdparty', 'bin')
    old_path = os.environ.get('PATH', '')
    if sys.platform == 'win32':
        #os.environ['PATH'] = thirdparty_dir + ';' + os.environ['PATH']
        os.environ['PATH'] = thirdparty_dir + ';' + old_path
    else:
        #os.environ['PATH'] = thirdparty_dir + ':' + os.environ['PATH']
        os.environ['PATH'] = thirdparty_dir + ':' + old_path
        os.environ['LD_LIBRARY_PATH'] = thirdparty_dir + ':' + os.environ.get('LD_LIBRARY_PATH', '')


def unfix_path():
    if sys.platform == 'win32':
        sep = ';'
    else:
        sep = ':'

    os.environ['PATH'] = sep.join(os.environ['PATH'].split(sep)[1:])


def get_available_languages():
    if getattr(sys, 'frozen', False):
        root_dir = os.path.dirname(os.path.dirname(sys.executable))
    else:
        path = os.path.abspath(__file__)
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    pretrained_dir = os.path.join(root_dir, 'pretrained_models')
    languages = []
    for f in os.listdir(pretrained_dir):
        if f.endswith(AcousticModel.extension):
            languages.append(os.path.splitext(f)[0])
    return languages


PRETRAINED_LANGUAGES = get_available_languages()


def align_corpus(args):
    all_begin = time.time()
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
                'type': 'align',
                'corpus_directory': args.corpus_directory,
                'dictionary_path': args.dictionary_path}
    if getattr(args, 'clean', False) \
            or conf['dirty'] or conf['type'] != 'align' \
            or conf['corpus_directory'] != args.corpus_directory \
            or conf['version'] != __version__ \
            or conf['dictionary_path'] != args.dictionary_path:
        shutil.rmtree(data_directory, ignore_errors=True)

    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(args.output_directory, exist_ok=True)
    try:
        corpus = Corpus(args.corpus_directory, data_directory,
                        speaker_characters=args.speaker_characters,
                        num_jobs=args.num_jobs,
                        ignore_exceptions=getattr(args, 'ignore_exceptions', False))
        if corpus.issues_check:
            print('WARNING: Some issues parsing the corpus were detected. '
                  'Please run the validator to get more information.')
        print(corpus.speaker_utterance_info())
        acoustic_model = AcousticModel(args.acoustic_model_path)
        dictionary = Dictionary(args.dictionary_path, data_directory, word_set=corpus.word_set)
        acoustic_model.validate(dictionary)

        begin = time.time()
        if args.config_path:
            align_config = align_yaml_to_config(args.config_path)
        else:
            align_config = load_basic_align()
        a = PretrainedAligner(corpus, dictionary, acoustic_model, align_config, args.output_directory,
                              temp_directory=data_directory,
                              debug=getattr(args, 'debug', False))
        if getattr(args, 'errors', False):
            check = a.test_utterance_transcriptions()
            if not getattr(args, 'quiet', False) and not check:
                user_input = input('Would you like to abort to fix transcription issues? (Y/N)')
                if user_input.lower() == 'y':
                    return
        if args.debug:
            print('Setup pretrained aligner in {} seconds'.format(time.time() - begin))
        a.verbose = args.verbose

        begin = time.time()
        a.align()
        if args.debug:
            print('Performed alignment in {} seconds'.format(time.time() - begin))

        begin = time.time()
        a.export_textgrids()
        if args.debug:
            print('Exported TextGrids in {} seconds'.format(time.time() - begin))
        print('Done! Everything took {} seconds'.format(time.time() - all_begin))
    except:
        conf['dirty'] = True
        raise
    finally:
        with open(conf_path, 'w') as f:
            yaml.dump(conf, f)


def align_included_model(args):
    if not os.path.exists(args.corpus_directory):
        raise (ArgumentError('Could not find the corpus directory {}.'.format(args.corpus_directory)))
    if not os.path.isdir(args.corpus_directory):
        raise (ArgumentError('The specified corpus directory ({}) is not a directory.'.format(args.corpus_directory)))
    if not os.path.exists(args.dictionary_path):
        raise (ArgumentError('Could not find the dictionary file {}'.format(args.dictionary_path)))
    if not os.path.isfile(args.dictionary_path):
        raise (ArgumentError('The specified dictionary path ({}) is not a text file.'.format(args.dictionary_path)))
    if getattr(sys, 'frozen', False):
        root_dir = os.path.dirname(os.path.dirname(sys.executable))
    else:
        path = os.path.abspath(__file__)
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    pretrained_dir = os.path.join(root_dir, 'pretrained_models')
    args.acoustic_model_path = os.path.join(pretrained_dir, '{}{}'.format(args.acoustic_model_path.lower(), AcousticModel.extension))
    align_corpus(args)


def validate_args(args):
    if args.acoustic_model_path.lower() in PRETRAINED_LANGUAGES:
        align_included_model(args)
    elif args.acoustic_model_path.lower().endswith(AcousticModel.extension):
        align_corpus(args)
    else:
        raise (Exception(
            'The language \'{}\' is not currently included in the distribution, '
            'please align via training or specify one of the following language names: {}.'.format(
                args.acoustic_model_path.lower(), ', '.join(PRETRAINED_LANGUAGES))))


if __name__ == '__main__':  # pragma: no cover
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_directory', help='Full path to the directory to align')
    parser.add_argument('dictionary_path', help='Full path to the pronunciation dictionary to use')
    parser.add_argument('acoustic_model_path',
                        help='Full path to the archive containing pre-trained model or language ({})'.format(
                            ', '.join(PRETRAINED_LANGUAGES)))
    parser.add_argument('output_directory', help="Full path to output directory, will be created if it doesn't exist")
    parser.add_argument('-s', '--speaker_characters', type=str, default='0',
                        help='Number of characters of file names to use for determining speaker, '
                             'default is to use directory names')
    parser.add_argument('-t', '--temp_directory', type=str, default='',
                        help='Temporary directory root to use for aligning, default is ~/Documents/MFA')
    parser.add_argument('-j', '--num_jobs', type=int, default=3,
                        help='Number of cores to use while aligning')
    parser.add_argument('-v', '--verbose', help="Print more information during alignment", action='store_true')
    parser.add_argument('-c', '--clean', help="Remove files from previous runs", action='store_true')
    parser.add_argument('-d', '--debug', help="Output debug messages about alignment", action='store_true')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to config file to use for alignment')
    args = parser.parse_args()
    try:
        args.speaker_characters = int(args.speaker_characters)
    except ValueError:
        pass

    args.output_directory = args.output_directory.rstrip('/').rstrip('\\')
    args.corpus_directory = args.corpus_directory.rstrip('/').rstrip('\\')

    if args.corpus_directory == args.output_directory:
        raise ArgumentError('Corpus directory and output directory cannot be the same folder.')

    fix_path()
    validate_args(args)
    unfix_path()
