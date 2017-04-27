import sys
import shutil, os
import time


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
    if sys.platform == 'win32':
        os.environ['PATH'] = thirdparty_dir + ';' + os.environ['PATH']
    else:
        os.environ['PATH'] = thirdparty_dir + ':' + os.environ['PATH']
        os.environ['LD_LIBRARY_PATH'] = thirdparty_dir + ':' + os.environ.get('LD_LIBRARY_PATH', '')


def unfix_path():
    if sys.platform == 'win32':
        sep = ';'
    else:
        sep = ':'

    os.environ['PATH'] = sep.join(os.environ['PATH'].split(sep)[1:])


fix_path()
import argparse
import multiprocessing as mp

from aligner.corpus import Corpus
from aligner.dictionary import Dictionary
from aligner.aligner import PretrainedAligner
from aligner.models import AcousticModel

PRETRAINED_LANGUAGES = ['english']

TEMP_DIR = os.path.expanduser('~/Documents/MFA')


def align_corpus(args, skip_input=False):
    all_begin = time.time()
    if not args.temp_directory:
        temp_dir = TEMP_DIR
    else:
        temp_dir = os.path.expanduser(args.temp_directory)
    corpus_name = os.path.basename(args.corpus_dir)
    if corpus_name == '':
        args.corpus_dir = os.path.dirname(args.corpus_dir)
        corpus_name = os.path.basename(args.corpus_dir)
    data_directory = os.path.join(temp_dir, corpus_name)
    if getattr(args, 'clean', False):
        shutil.rmtree(data_directory, ignore_errors=True)
        shutil.rmtree(args.output_directory, ignore_errors=True)

    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(args.output_directory, exist_ok=True)
    use_speaker_info = not args.no_speaker_adaptation
    corpus = Corpus(args.corpus_dir, data_directory,
                    speaker_characters=args.speaker_characters,
                    num_jobs=args.num_jobs,
                    use_speaker_information=use_speaker_info,
                    ignore_exceptions=getattr(args, 'ignore_exceptions', False))
    print(corpus.speaker_utterance_info())
    acoustic_model = AcousticModel(args.model_path)
    dictionary = Dictionary(args.dict_path, data_directory, word_set=corpus.word_set)
    acoustic_model.validate(dictionary)
    begin = time.time()
    a = PretrainedAligner(corpus, dictionary, acoustic_model, args.output_directory, temp_directory=data_directory,
                          num_jobs=getattr(args, 'num_jobs', 3),
                          speaker_independent=getattr(args, 'no_speaker_adaptation', False),
                          debug=getattr(args, 'debug', False))
    if getattr(args, 'errors', False):
        check = a.test_utterance_transcriptions()
        if not skip_input and not check:
            user_input = input('Would you like to abort to fix transcription issues? (Y/N)')
            if user_input.lower() == 'y':
                return
    if args.debug:
        print('Setup pretrained aligner in {} seconds'.format(time.time() - begin))
    a.verbose = args.verbose
    utt_oov_path = os.path.join(corpus.split_directory, 'utterance_oovs.txt')
    if os.path.exists(utt_oov_path):
        shutil.copy(utt_oov_path, args.output_directory)
    oov_path = os.path.join(corpus.split_directory, 'oovs_found.txt')
    if os.path.exists(oov_path):
        shutil.copy(oov_path, args.output_directory)
    if not skip_input and a.dictionary.oovs_found:
        user_input = input('There were words not found in the dictionary. Would you like to abort to fix them? (Y/N)')
        if user_input.lower() == 'y':
            return
    begin = time.time()
    a.do_align()
    if args.debug:
        print('Performed alignment in {} seconds'.format(time.time() - begin))
    begin = time.time()
    a.export_textgrids()
    if args.debug:
        print('Exported textgrids in {} seconds'.format(time.time() - begin))
    print('Done! Everything took {} seconds'.format(time.time() - all_begin))


def align_included_model(args, skip_input=False):
    if getattr(sys, 'frozen', False):
        root_dir = os.path.dirname(os.path.dirname(sys.executable))
    else:
        path = os.path.abspath(__file__)
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    pretrained_dir = os.path.join(root_dir, 'pretrained_models')
    args.model_path = os.path.join(pretrained_dir, '{}.zip'.format(args.model_path.lower()))
    align_corpus(args, skip_input=skip_input)

def validate_args(args):

    if args.model_path.lower() in PRETRAINED_LANGUAGES:
        align_included_model(args)
    elif args.model_path.lower().endswith('.zip'):
        align_corpus(args)
    else:
        raise (Exception(
            'The language \'{}\' is not currently included in the distribution, '
            'please align via training or specify one of the following language names: {}.'.format(
                args.model_path.lower(), ', '.join(PRETRAINED_LANGUAGES))))

if __name__ == '__main__':  # pragma: no cover
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('dictionary', help='Full path to the pronunciation dictionary to use')
    parser.add_argument('acoustic_model',
                        help='Full path to the archive containing pre-trained model or language ({})'.format(
                            ', '.join(PRETRAINED_LANGUAGES)))
    parser.add_argument('corpus_directory', help='Full path to the directory to align')
    parser.add_argument('output_directory', help="Full path to output directory, will be created if it doesn't exist")
    parser.add_argument('-s', '--speaker_characters', type=str, default='0',
                        help='Number of characters of file names to use for determining speaker, '
                             'default is to use directory names')
    parser.add_argument('-t', '--temp_directory', type=str, default='',
                        help='Temporary directory root to use for aligning, default is ~/Documents/MFA')
    parser.add_argument('-j', '--num_jobs', type=int, default=3,
                        help='Number of cores to use while aligning')
    parser.add_argument('-v', '--verbose', help="Output debug messages about alignment", action='store_true')
    parser.add_argument('-n', '--no_speaker_adaptation',
                        help="Only use speaker independent models, with no speaker adaptation", action='store_true')
    parser.add_argument('-c', '--clean', help="Remove files from previous runs", action='store_true')
    parser.add_argument('-d', '--debug', help="Debug the aligner", action='store_true')
    parser.add_argument('-e', '--errors', help="Test for transcription errors in files to be aligned",
                        action='store_true')
    parser.add_argument('-i', '--ignore_exceptions', help='Ignore exceptions raised when parsing data',
                        action='store_true')
    args = parser.parse_args()
    try:
        args.speaker_characters = int(args.speaker_characters)
    except ValueError:
        pass
    validate_args(args)
    unfix_path()
