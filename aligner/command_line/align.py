import sys
import shutil, os
import time


def fix_path():
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
        if sys.platform == 'win32':
            thirdparty_dir = os.path.join(base_dir, 'thirdparty', 'bin')
        else:
            thirdparty_dir = os.path.join(base_dir, 'thirdparty','bin')
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
from aligner.aligner import PretrainedAligner
from aligner.archive import Archive

PRETRAINED_LANGUAGES = ['english']

TEMP_DIR = os.path.expanduser('~/Documents/MFA')


def align_corpus(model_path, corpus_dir,  output_directory, temp_dir, args, debug=False):
    all_begin = time.time()
    if temp_dir == '':
        temp_dir = TEMP_DIR
    else:
        temp_dir = os.path.expanduser(temp_dir)
    corpus_name = os.path.basename(corpus_dir)
    if corpus_name == '':
        corpus_dir = os.path.dirname(corpus_dir)
        corpus_name = os.path.basename(corpus_dir)
    data_directory = os.path.join(temp_dir, corpus_name)
    if getattr(args,'clean',False):
        shutil.rmtree(data_directory, ignore_errors=True)
        shutil.rmtree(output_directory, ignore_errors=True)

    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(output_directory, exist_ok=True)
    begin = time.time()
    use_speaker_info = not args.no_speaker_adaptation
    corpus = Corpus(corpus_dir, data_directory,
                    speaker_characters=args.speaker_characters,
                    num_jobs=args.num_jobs,
                    use_speaker_information=use_speaker_info)
    print(corpus.speaker_utterance_info())
    corpus.write()
    if debug:
        print('Wrote corpus information in {} seconds'.format(time.time() - begin))
    begin = time.time()
    corpus.create_mfccs()
    if debug:
        print('Calculated mfccs in {} seconds'.format(time.time() - begin))
    archive = Archive(model_path)
    begin = time.time()
    a = PretrainedAligner(archive, corpus, output_directory, temp_directory=data_directory,
                          num_jobs = getattr(args, 'num_jobs',3), speaker_independent=getattr(args, 'no_speaker_adaptation',False),
                          debug=getattr(args, 'debug', False))
    if getattr(args, 'errors', False):

        check = a.test_utterance_transcriptions()
        if not check:
            user_input = input('Would you like to abort to fix transcription issues? (Y/N)')
            if user_input.lower() == 'y':
                return
    if debug:
        print('Setup pretrained aligner in {} seconds'.format(time.time() - begin))
    a.verbose = args.verbose
    begin = time.time()
    corpus.setup_splits(a.dictionary)
    if debug:
        print('Setup splits in {} seconds'.format(time.time() - begin))
    utt_oov_path = os.path.join(corpus.split_directory, 'utterance_oovs.txt')
    if os.path.exists(utt_oov_path):
        shutil.copy(utt_oov_path, output_directory)
    oov_path = os.path.join(corpus.split_directory, 'oovs_found.txt')
    if os.path.exists(oov_path):
        shutil.copy(oov_path, output_directory)
    begin = time.time()
    a.do_align()
    if debug:
        print('Performed alignment in {} seconds'.format(time.time() - begin))
    begin = time.time()
    a.export_textgrids()
    if debug:
        print('Exported textgrids in {} seconds'.format(time.time() - begin))
    print('Done! Everything took {} seconds'.format(time.time() - all_begin))


def align_included_model(language, corpus_dir,  output_directory, temp_dir, args):
    if language not in PRETRAINED_LANGUAGES:
        raise(Exception('The language \'{}\' is not currently included in the distribution, please align via training or specify one of the following language names: {}.'.format(language, ', '.join(PRETRAINED_LANGUAGES))))

    if getattr(sys, 'frozen', False):
        root_dir = os.path.dirname(os.path.dirname(sys.executable))
    else:
        path = os.path.abspath(__file__)
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    pretrained_dir = os.path.join(root_dir, 'pretrained_models')
    model_path = os.path.join(pretrained_dir, '{}.zip'.format(language))
    align_corpus(model_path, corpus_dir,  output_directory, temp_dir, args)


if __name__ == '__main__':  # pragma: no cover
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', nargs = '?', help = 'Full path to the archive containing pre-trained model', default = '')
    parser.add_argument('corpus_dir', help = 'Full path to the directory to align')
    parser.add_argument('output_dir', help = 'Full path to output directory, will be created if it doesn\'t exist')
    parser.add_argument('-s', '--speaker_characters', type = int, default = 0,
                    help = 'Number of characters of filenames to use for determining speaker, default is to use directory names')
    parser.add_argument('-t', '--temp_directory', type = str, default = '',
                    help = 'Temporary directory root to use for aligning, default is ~/Documents/MFA')
    parser.add_argument('-j','--num_jobs', type = int, default = 3,
                    help = 'Number of cores to use while aligning')
    parser.add_argument('-v', '--verbose', help = "Output debug messages about alignment", action = 'store_true')
    parser.add_argument('--language', type = str, default = '',
                    help = 'Specify whether to use an included pretrained model (english, french)')
    parser.add_argument('-n', '--no_speaker_adaptation', help = "Only use speaker independent models, with no speaker adaptation", action = 'store_true')
    parser.add_argument('-c', '--clean', help = "Remove files from previous runs", action = 'store_true')
    parser.add_argument('-d', '--debug', help = "Debug the aligner", action = 'store_true')
    parser.add_argument('-e', '--errors', help = "Test for transcription errors in files to be aligned", action = 'store_true')
    args = parser.parse_args()
    corpus_dir = os.path.expanduser(args.corpus_dir)
    model_path = os.path.expanduser(args.model_path)
    output_dir = os.path.expanduser(args.output_dir)
    language = args.language.lower()
    temp_dir = args.temp_directory
    if language == '' and model_path == '':
        raise(Exception('Both language and model_path cannot be unspecified'))
    elif language != '' and model_path != '':
        raise(Exception('Both language and model_path cannot be specified'))
    if model_path != '':
        align_corpus(model_path, corpus_dir, output_dir, temp_dir,
            args)
    else:
        align_included_model(language, corpus_dir, output_dir, temp_dir,
            args)
    unfix_path()
