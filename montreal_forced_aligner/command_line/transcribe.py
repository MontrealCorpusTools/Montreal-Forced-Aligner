import shutil
import os
import time
import multiprocessing as mp
import yaml

from montreal_forced_aligner import __version__
from montreal_forced_aligner.corpus import AlignableCorpus, TranscribeCorpus
from montreal_forced_aligner.dictionary import Dictionary
from montreal_forced_aligner.transcriber import Transcriber
from montreal_forced_aligner.models import AcousticModel, LanguageModel
from montreal_forced_aligner.config import TEMP_DIR, transcribe_yaml_to_config, load_basic_transcribe, save_config
from montreal_forced_aligner.utils import get_available_acoustic_languages, get_pretrained_acoustic_path, \
    get_available_lm_languages, get_pretrained_language_model_path
from montreal_forced_aligner.exceptions import ArgumentError


class DummyArgs(object):
    def __init__(self):
        self.corpus_directory = ''
        self.dictionary_path = ''
        self.acoustic_model_path = ''
        self.speaker_characters = 0
        self.num_jobs = 0
        self.verbose = False
        self.clean = True
        self.fast = True
        self.debug = False
        self.temp_directory = None
        self.config_path = ''


def transcribe_corpus(args):
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
    print(data_directory, os.path.exists(data_directory))
    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(args.output_directory, exist_ok=True)
    os.makedirs(data_directory, exist_ok=True)
    conf_path = os.path.join(data_directory, 'config.yml')
    if os.path.exists(conf_path):
        with open(conf_path, 'r') as f:
            conf = yaml.load(f, Loader=yaml.SafeLoader)
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
        pass  # FIXME
        # shutil.rmtree(data_directory, ignore_errors=True)
    try:
        if args.evaluate:
            corpus = AlignableCorpus(args.corpus_directory, data_directory,
                                     speaker_characters=args.speaker_characters,
                                     num_jobs=args.num_jobs)
        else:
            corpus = TranscribeCorpus(args.corpus_directory, data_directory,
                                      speaker_characters=args.speaker_characters,
                                      num_jobs=args.num_jobs)
        print(corpus.speaker_utterance_info())
        acoustic_model = AcousticModel(args.acoustic_model_path)
        language_model = LanguageModel(args.language_model_path)
        dictionary = Dictionary(args.dictionary_path, data_directory)
        acoustic_model.validate(dictionary)

        if args.config_path:
            transcribe_config = transcribe_yaml_to_config(args.config_path)
        else:
            transcribe_config = load_basic_transcribe()
        begin = time.time()
        t = Transcriber(corpus, dictionary, acoustic_model, language_model, transcribe_config,
                        temp_directory=data_directory,
                        debug=getattr(args, 'debug', False), evaluation_mode=args.evaluate)
        if args.debug:
            print('Setup pretrained aligner in {} seconds'.format(time.time() - begin))

        begin = time.time()
        t.transcribe()
        if args.debug:
            print('Performed transcribing in {} seconds'.format(time.time() - begin))
        if args.evaluate:
            t.evaluate(args.output_directory)
            best_config_path = os.path.join(args.output_directory, 'best_transcribe_config.yaml')
            save_config(t.transcribe_config, best_config_path)
            t.export_transcriptions(args.output_directory)
        else:
            begin = time.time()
            t.export_transcriptions(args.output_directory)
            if args.debug:
                print('Exported transcriptions in {} seconds'.format(time.time() - begin))
        print('Done! Everything took {} seconds'.format(time.time() - all_begin))
    except Exception as _:
        conf['dirty'] = True
        raise
    finally:
        if os.path.exists(data_directory):
            with open(conf_path, 'w') as f:
                yaml.dump(conf, f)


def validate_args(args, pretrained_acoustic, pretrained_lm):
    if not os.path.exists(args.corpus_directory):
        raise ArgumentError('Could not find the corpus directory {}.'.format(args.corpus_directory))
    if not os.path.isdir(args.corpus_directory):
        raise ArgumentError('The specified corpus directory ({}) is not a directory.'.format(args.corpus_directory))
    if not os.path.exists(args.dictionary_path):
        raise ArgumentError('Could not find the dictionary file {}'.format(args.dictionary_path))
    if not os.path.isfile(args.dictionary_path):
        raise ArgumentError('The specified dictionary path ({}) is not a text file.'.format(args.dictionary_path))
    if args.corpus_directory == args.output_directory:
        raise ArgumentError('Corpus directory and output directory cannot be the same folder.')

    if args.acoustic_model_path.lower() in pretrained_acoustic:
        args.acoustic_model_path = get_pretrained_acoustic_path(args.acoustic_model_path.lower())
    elif args.acoustic_model_path.lower().endswith(AcousticModel.extension):
        if not os.path.exists(args.acoustic_model_path):
            raise ArgumentError('The specified model path does not exist: ' + args.acoustic_model_path)
    else:
        raise ArgumentError(
            'The acoustic model \'{}\' is not currently downloaded, '
            'please download a pretrained model, align via training or specify one of the following language names: {}.'.format(
                args.acoustic_model_path.lower(), ', '.join(pretrained_acoustic)))

    if args.language_model_path.lower() in pretrained_lm:
        args.language_model_path = get_pretrained_language_model_path(args.language_model_path.lower())
    elif args.language_model_path.lower().endswith(LanguageModel.extension):
        if not os.path.exists(args.language_model_path):
            raise ArgumentError('The specified model path does not exist: ' + args.language_model_path)
    else:
        raise ArgumentError(
            'The language model \'{}\' is not currently downloaded, '
            'please download, train a new model, or specify one of the following language names: {}.'.format(
                args.language_model_path.lower(), ', '.join(pretrained_lm)))


def run_transcribe_corpus(args, pretrained_acoustic=None, pretrained_lm=None):
    if pretrained_acoustic is None:
        pretrained_acoustic = get_available_acoustic_languages()
    if pretrained_lm is None:
        pretrained_lm = get_available_lm_languages()
    try:
        args.speaker_characters = int(args.speaker_characters)
    except ValueError:
        pass
    args.output_directory = args.output_directory.rstrip('/').rstrip('\\')
    args.corpus_directory = args.corpus_directory.rstrip('/').rstrip('\\')

    validate_args(args, pretrained_acoustic, pretrained_lm)
    transcribe_corpus(args)


if __name__ == '__main__':  # pragma: no cover
    mp.freeze_support()
    from montreal_forced_aligner.command_line.mfa import transcribe_parser, fix_path, unfix_path, acoustic_languages, \
        lm_languages

    transcribe_args = transcribe_parser.parse_args()
    fix_path()
    run_transcribe_corpus(transcribe_args, acoustic_languages, lm_languages)
    unfix_path()
