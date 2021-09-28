import os
import time
import shutil
import multiprocessing as mp

from montreal_forced_aligner.corpus.align_corpus import AlignableCorpus
from montreal_forced_aligner.dictionary import Dictionary

from montreal_forced_aligner.config import TEMP_DIR, train_lm_yaml_to_config, load_basic_train_lm

from montreal_forced_aligner.exceptions import ArgumentError

from montreal_forced_aligner.lm.trainer import LmTrainer
from montreal_forced_aligner.utils import get_available_dict_languages, get_dictionary_path
from montreal_forced_aligner.helper import setup_logger


def train_lm(args, unknown_args=None):
    command = 'train_lm'
    all_begin = time.time()
    if not args.temp_directory:
        temp_dir = TEMP_DIR
    else:
        temp_dir = os.path.expanduser(args.temp_directory)
    if args.config_path:
        train_config = train_lm_yaml_to_config(args.config_path)
    else:
        train_config = load_basic_train_lm()
    train_config.use_mp = not args.disable_mp
    if unknown_args:
        train_config.update_from_args(unknown_args)
    corpus_name = os.path.basename(args.source_path)
    if corpus_name == '':
        args.source_path = os.path.dirname(args.source_path)
        corpus_name = os.path.basename(args.source_path)
    source = args.source_path
    dictionary = None
    if args.source_path.lower().endswith('.arpa'):
        corpus_name = os.path.splitext(corpus_name)[0]
        data_directory = os.path.join(temp_dir, corpus_name)
    else:
        data_directory = os.path.join(temp_dir, corpus_name)
    if getattr(args, 'clean', False) and os.path.exists(data_directory):
        print('Cleaning old directory!')
        shutil.rmtree(data_directory, ignore_errors=True)

    if getattr(args, 'verbose', False):
        log_level = 'debug'
    else:
        log_level = 'info'
    logger = setup_logger(command, data_directory, console_level=log_level)
    if not args.source_path.lower().endswith('.arpa'):
        source = AlignableCorpus(args.source_path, data_directory, num_jobs=args.num_jobs, use_mp=train_config.use_mp,
                                 parse_text_only_files=True, debug=args.debug)
        if args.dictionary_path is not None:
            dictionary = Dictionary(args.dictionary_path, data_directory, debug=args.debug, word_set=source.word_set)
            dictionary.generate_mappings()
        else:
            dictionary = None
    trainer = LmTrainer(source, train_config, args.output_model_path, dictionary=dictionary,
                        temp_directory=data_directory,
                        supplemental_model_path=args.model_path, supplemental_model_weight=args.model_weight, debug=args.debug, logger=logger)
    begin = time.time()
    trainer.train()
    logger.debug('Training took {} seconds'.format(time.time() - begin))

    logger.info('All done!')
    logger.debug('Done! Everything took {} seconds'.format(time.time() - all_begin))
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


def validate_args(args, download_dictionaries=None):
    if args.dictionary_path is not None and args.dictionary_path.lower() in download_dictionaries:
        args.dictionary_path = get_dictionary_path(args.dictionary_path.lower())
    if not args.source_path.endswith('.arpa'):
        if not os.path.exists(args.source_path):
            raise (ArgumentError('Could not find the corpus directory {}.'.format(args.source_path)))
        if not os.path.isdir(args.source_path):
            raise (ArgumentError('The specified corpus directory ({}) is not a directory.'.format(args.source_path)))
    else:
        if not os.path.exists(args.source_path):
            raise (ArgumentError('Could not find the source file {}.'.format(args.source_path)))
    if args.config_path and not os.path.exists(args.config_path):
        raise (ArgumentError('Could not find the config file {}.'.format(args.config_path)))
    if args.model_path and not os.path.exists(args.model_path):
        raise (ArgumentError('Could not find the model file {}.'.format(args.model_path)))


def run_train_lm(args, unknown=None, download_dictionaries=None):
    if not args.dictionary_path:
        args.dictionary_path = None
    if download_dictionaries is None:
        download_dictionaries = get_available_dict_languages()
    args.source_path = args.source_path.rstrip('/').rstrip('\\')

    validate_args(args, download_dictionaries)
    train_lm(args, unknown)

