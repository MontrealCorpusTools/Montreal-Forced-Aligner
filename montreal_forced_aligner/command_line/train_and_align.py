import shutil
import os
import multiprocessing as mp
import yaml
import time

from montreal_forced_aligner import __version__
from montreal_forced_aligner.corpus.align_corpus import AlignableCorpus
from montreal_forced_aligner.dictionary import Dictionary
from montreal_forced_aligner.aligner import TrainableAligner
from montreal_forced_aligner.config import TEMP_DIR, train_yaml_to_config, load_basic_train
from montreal_forced_aligner.utils import get_available_dict_languages, get_dictionary_path

from montreal_forced_aligner.exceptions import ArgumentError


def align_corpus(args):
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
            conf = yaml.load(f, Loader=yaml.SafeLoader)
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

    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(args.output_directory, exist_ok=True)
    try:
        corpus = AlignableCorpus(args.corpus_directory, data_directory, speaker_characters=args.speaker_characters,
                        num_jobs=getattr(args, 'num_jobs', 3),
                        debug=getattr(args, 'debug', False))
        if corpus.issues_check:
            print('WARNING: Some issues parsing the corpus were detected. '
                  'Please run the validator to get more information.')
        dictionary = Dictionary(args.dictionary_path, data_directory, word_set=corpus.word_set)
        utt_oov_path = os.path.join(corpus.split_directory(), 'utterance_oovs.txt')
        if os.path.exists(utt_oov_path):
            shutil.copy(utt_oov_path, args.output_directory)
        oov_path = os.path.join(corpus.split_directory(), 'oovs_found.txt')
        if os.path.exists(oov_path):
            shutil.copy(oov_path, args.output_directory)
        if args.config_path:
            train_config, align_config = train_yaml_to_config(args.config_path)
        else:
            train_config, align_config = load_basic_train()
        a = TrainableAligner(corpus, dictionary, train_config, align_config,
                             temp_directory=data_directory)
        a.verbose = args.verbose
        a.train()
        a.export_textgrids(args.output_directory)
        if args.output_model_path is not None:
            a.save(args.output_model_path)
    except Exception as _:
        conf['dirty'] = True
        raise
    finally:
        with open(conf_path, 'w') as f:
            yaml.dump(conf, f)


def validate_args(args, download_dictionaries):
    if args.corpus_directory == args.output_directory:
        raise Exception('Corpus directory and output directory cannot be the same folder.')
    if not os.path.exists(args.corpus_directory):
        raise (ArgumentError('Could not find the corpus directory {}.'.format(args.corpus_directory)))
    if not os.path.isdir(args.corpus_directory):
        raise (ArgumentError('The specified corpus directory ({}) is not a directory.'.format(args.corpus_directory)))

    if args.dictionary_path.lower() in download_dictionaries:
        args.dictionary_path = get_dictionary_path(args.dictionary_path.lower())
    if not os.path.exists(args.dictionary_path):
        raise (ArgumentError('Could not find the dictionary file {}'.format(args.dictionary_path)))
    if not os.path.isfile(args.dictionary_path):
        raise (ArgumentError('The specified dictionary path ({}) is not a text file.'.format(args.dictionary_path)))


def run_train_corpus(args, download_dictionaries=None):
    if download_dictionaries is None:
        download_dictionaries = get_available_dict_languages()
    try:
        args.speaker_characters = int(args.speaker_characters)
    except ValueError:
        pass
    if not args.output_model_path:
        args.output_model_path = None
    args.output_directory = args.output_directory.rstrip('/').rstrip('\\')
    args.corpus_directory = args.corpus_directory.rstrip('/').rstrip('\\')

    validate_args(args, download_dictionaries)
    align_corpus(args)


if __name__ == '__main__':  # pragma: no cover
    mp.freeze_support()
    from montreal_forced_aligner.command_line.mfa import train_parser, fix_path, unfix_path, dict_languages
    train_args = train_parser.parse_args()

    fix_path()
    run_train_corpus(train_args, dict_languages)
    unfix_path()
