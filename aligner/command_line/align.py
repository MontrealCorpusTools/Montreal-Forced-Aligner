import sys
import shutil, os


def fix_path():
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
        if sys.platform == 'win32':
            thirdparty_dir = os.path.join(base_dir, 'thirdparty', 'bin')
        else:
            thirdparty_dir = os.path.join(base_dir, 'lib', 'thirdparty','bin')
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        thirdparty_dir = os.path.join(base_dir, 'thirdparty', 'bin')
    if sys.platform == 'win32':
        os.environ['PATH'] = thirdparty_dir + ';' + os.environ['PATH']
    else:
        os.environ['PATH'] = thirdparty_dir + ':' + os.environ['PATH']

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
from aligner.archive import Archive

PRETRAINED_LANGUAGES = ['english']

TEMP_DIR = os.path.expanduser('~/Documents/MFA')

def align_corpus(model_path, corpus_dir,  output_directory, speaker_characters, num_jobs, verbose, clean):

    corpus_name = os.path.basename(corpus_dir)
    if corpus_name == '':
        corpus_dir = os.path.dirname(corpus_dir)
        corpus_name = os.path.basename(corpus_dir)
    data_directory = os.path.join(TEMP_DIR, corpus_name)
    if clean:
        shutil.rmtree(data_directory, ignore_errors = True)
        shutil.rmtree(output_directory, ignore_errors = True)

    os.makedirs(data_directory, exist_ok = True)
    os.makedirs(output_directory, exist_ok = True)
    corpus = Corpus(corpus_dir, data_directory, speaker_characters, num_jobs = num_jobs)
    print(corpus.speaker_utterance_info())
    corpus.write()
    corpus.create_mfccs()
    archive = Archive(model_path)
    a = PretrainedAligner(archive, corpus, output_directory,
                        temp_directory = data_directory, num_jobs = num_jobs)
    a.verbose = verbose
    corpus.setup_splits(a.dictionary)
    utt_oov_path = os.path.join(corpus.split_directory, 'utterance_oovs.txt')
    if os.path.exists(utt_oov_path):
        shutil.copy(utt_oov_path, output_directory)
    oov_path = os.path.join(corpus.split_directory, 'oovs_found.txt')
    if os.path.exists(oov_path):
        shutil.copy(oov_path, output_directory)
    a.do_align()
    a.export_textgrids()

def align_included_model(language, corpus_dir,  output_directory,
                        speaker_characters, num_jobs, verbose, clean):
    if language not in PRETRAINED_LANGUAGES:
        raise(Exception('The language \'{}\' is not currently included in the distribution, please align via training or specify one of the following language names: {}.'.format(language, ', '.join(PRETRAINED_LANGUAGES))))

    if getattr(sys, 'frozen', False):
        root_dir = os.path.dirname(os.path.dirname(sys.executable))
    else:
        path = os.path.abspath(__file__)
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    pretrained_dir = os.path.join(root_dir, 'pretrained_models')
    model_path = os.path.join(pretrained_dir, '{}.zip'.format(language))
    align_corpus(model_path, corpus_dir,  output_directory, speaker_characters, num_jobs, verbose, clean)

if __name__ == '__main__': # pragma: no cover
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', nargs = '?', help = 'Full path to the archive containing pre-trained model', default = '')
    parser.add_argument('corpus_dir', help = 'Full path to the directory to align')
    parser.add_argument('output_dir', help = 'Full path to output directory, will be created if it doesn\'t exist')
    parser.add_argument('-s', '--speaker_characters', type = int, default = 0,
                    help = 'Number of characters of filenames to use for determining speaker, default is to use directory names')
    parser.add_argument('-j','--num_jobs', type = int, default = 3,
                    help = 'Number of cores to use while aligning')
    parser.add_argument('-v', '--verbose', help = "Output debug messages about alignment", action = 'store_true')
    parser.add_argument('--language', type = str, default = '',
                    help = 'Specify whether to use an included pretrained model (english, french)')
    parser.add_argument('-c', '--clean', help = "Remove files from previous runs", action = 'store_true')
    args = parser.parse_args()
    corpus_dir = os.path.expanduser(args.corpus_dir)
    model_path = os.path.expanduser(args.model_path)
    output_dir = os.path.expanduser(args.output_dir)
    language = args.language.lower()
    if language == '' and model_path == '':
        raise(Exception('Both language and model_path cannot be unspecified'))
    elif language != '' and model_path != '':
        raise(Exception('Both language and model_path cannot be specified'))
    if model_path != '':
        align_corpus(model_path, corpus_dir, output_dir,
            args.speaker_characters, args.num_jobs, args.verbose, args.clean)
    else:
        align_included_model(language, corpus_dir, output_dir,
            args.speaker_characters, args.num_jobs, args.verbose, args.clean)
    unfix_path()
