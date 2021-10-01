import shutil
import os
import time
import multiprocessing as mp

from montreal_forced_aligner.corpus.align_corpus import AlignableCorpus
from montreal_forced_aligner.dictionary import Dictionary
from montreal_forced_aligner.validator import CorpusValidator
from montreal_forced_aligner.exceptions import ArgumentError
from montreal_forced_aligner.config import TEMP_DIR
from montreal_forced_aligner.utils import get_available_acoustic_languages, get_pretrained_acoustic_path, \
    get_available_dict_languages, get_dictionary_path
from montreal_forced_aligner.helper import setup_logger
from montreal_forced_aligner.models import AcousticModel


def validate_corpus(args, unknown_args=None):
    command = 'validate'
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
    shutil.rmtree(data_directory, ignore_errors=True)

    os.makedirs(data_directory, exist_ok=True)
    model_directory = os.path.join(data_directory, 'acoustic_models')
    os.makedirs(model_directory, exist_ok=True)
    if getattr(args, 'verbose', False):
        log_level = 'debug'
    else:
        log_level = 'info'
    logger = setup_logger(command, data_directory, console_level=log_level)

    corpus = AlignableCorpus(args.corpus_directory, data_directory, speaker_characters=args.speaker_characters,
                    num_jobs=getattr(args, 'num_jobs', 3), logger=logger, use_mp=not args.disable_mp)
    if args.acoustic_model_path:
        acoustic_model = AcousticModel(args.acoustic_model_path, root_directory=model_directory)
        acoustic_model.log_details(logger)
        dictionary = Dictionary(args.dictionary_path, data_directory, logger=logger, multilingual_ipa=acoustic_model.meta['multilingual_ipa'])
        acoustic_model.validate(dictionary)
    else:
        dictionary = Dictionary(args.dictionary_path, data_directory, logger=logger)

    a = CorpusValidator(corpus, dictionary, temp_directory=data_directory,
                        ignore_acoustics=getattr(args, 'ignore_acoustics', False),
                        test_transcriptions=getattr(args, 'test_transcriptions', False), use_mp=not args.disable_mp,
                        logger=logger)
    begin = time.time()
    a.validate()
    logger.debug('Validation took {} seconds'.format(time.time() - begin))
    logger.info('All done!')
    logger.debug('Done! Everything took {} seconds'.format(time.time() - all_begin))
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


def validate_args(args, downloaded_acoustic_models=None, download_dictionaries=None):
    if args.test_transcriptions and args.ignore_acoustics:
        raise ArgumentError('Cannot test transcriptions without acoustic feature generation.')
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

    if args.acoustic_model_path:
        if args.acoustic_model_path.lower() in downloaded_acoustic_models:
            args.acoustic_model_path = get_pretrained_acoustic_path(args.acoustic_model_path.lower())
        elif args.acoustic_model_path.lower().endswith(AcousticModel.extension):
            if not os.path.exists(args.acoustic_model_path):
                raise ArgumentError('The specified model path does not exist: ' + args.acoustic_model_path)
        else:
            raise ArgumentError(
                'The language \'{}\' is not currently included in the distribution, '
                'please align via training or specify one of the following language names: {}.'.format(
                    args.acoustic_model_path.lower(), ', '.join(downloaded_acoustic_models)))


def run_validate_corpus(args, unknown=None, downloaded_acoustic_models=None, download_dictionaries=None):
    if downloaded_acoustic_models is None:
        downloaded_acoustic_models = get_available_acoustic_languages()
    if download_dictionaries is None:
        download_dictionaries = get_available_dict_languages()
    try:
        args.speaker_characters = int(args.speaker_characters)
    except ValueError:
        pass
    validate_args(args, downloaded_acoustic_models, download_dictionaries)
    validate_corpus(args, unknown)

