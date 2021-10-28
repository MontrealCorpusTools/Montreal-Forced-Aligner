from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Collection
if TYPE_CHECKING:
    from ..corpus import AlignableCorpus
    from ..dictionary import Dictionary
    from argparse import Namespace
import shutil
import os
import time

from montreal_forced_aligner.corpus.align_corpus import AlignableCorpus
from montreal_forced_aligner.dictionary import Dictionary
from montreal_forced_aligner.validator import CorpusValidator
from montreal_forced_aligner.exceptions import ArgumentError
from montreal_forced_aligner.config import TEMP_DIR
from montreal_forced_aligner.command_line.utils import validate_model_arg
from montreal_forced_aligner.utils import setup_logger
from montreal_forced_aligner.models import AcousticModel


def validate_corpus(args: Namespace, unknown_args: Optional[list]=None) -> None:
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


def validate_args(args: Namespace) -> None:
    try:
        args.speaker_characters = int(args.speaker_characters)
    except ValueError:
        pass
    if args.test_transcriptions and args.ignore_acoustics:
        raise ArgumentError('Cannot test transcriptions without acoustic feature generation.')
    if not os.path.exists(args.corpus_directory):
        raise (ArgumentError('Could not find the corpus directory {}.'.format(args.corpus_directory)))
    if not os.path.isdir(args.corpus_directory):
        raise (ArgumentError('The specified corpus directory ({}) is not a directory.'.format(args.corpus_directory)))

    args.dictionary_path = validate_model_arg(args.dictionary_path, 'dictionary')
    if args.acoustic_model_path:
        args.acoustic_model_path = validate_model_arg(args.acoustic_model_path, 'acoustic')


def run_validate_corpus(args: Namespace, unknown: Optional[list]=None) -> None:
    validate_args(args)
    validate_corpus(args, unknown)

