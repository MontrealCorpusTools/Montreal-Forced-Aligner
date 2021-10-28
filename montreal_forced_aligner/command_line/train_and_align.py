from __future__ import annotations
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from argparse import Namespace
import shutil
import os
import time

from montreal_forced_aligner import __version__
from montreal_forced_aligner.corpus.align_corpus import AlignableCorpus
from montreal_forced_aligner.dictionary import Dictionary
from montreal_forced_aligner.aligner import TrainableAligner
from montreal_forced_aligner.config import TEMP_DIR, train_yaml_to_config, load_basic_train, load_command_configuration
from montreal_forced_aligner.command_line.utils import validate_model_arg
from montreal_forced_aligner.utils import setup_logger, log_config

from montreal_forced_aligner.exceptions import ArgumentError


def align_corpus(args: Namespace, unknown_args: Optional[list]=None) -> None:
    command = 'train_and_align'
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
    if args.config_path:
        train_config, align_config = train_yaml_to_config(args.config_path)
    else:
        train_config, align_config = load_basic_train()
    train_config.use_mp = not args.disable_mp
    align_config.use_mp = not args.disable_mp
    align_config.debug = args.debug
    align_config.overwrite = args.overwrite
    align_config.cleanup_textgrids = not args.disable_textgrid_cleanup
    if unknown_args:
        train_config.update_from_unknown_args(unknown_args)
        align_config.update_from_unknown_args(unknown_args)
    train_config.update_from_align(align_config)
    conf_path = os.path.join(data_directory, 'config.yml')
    if getattr(args, 'clean', False) and os.path.exists(data_directory):
        print('Cleaning old directory!')
        shutil.rmtree(data_directory, ignore_errors=True)
    if getattr(args, 'verbose', False):
        log_level = 'debug'
    else:
        log_level = 'info'
    logger = setup_logger(command, data_directory, console_level=log_level)
    logger.debug('TRAIN CONFIG:')
    log_config(logger, train_config)
    logger.debug('ALIGN CONFIG:')
    log_config(logger, align_config)
    if args.debug:
        logger.warning('Running in DEBUG mode, may have impact on performance and disk usage.')
    conf = load_command_configuration(conf_path, {'dirty': False,
                'begin': time.time(),
                'version': __version__,
                'type': command,
                'corpus_directory': args.corpus_directory,
                'dictionary_path': args.dictionary_path})
    if conf['dirty'] or conf['type'] != command \
            or conf['corpus_directory'] != args.corpus_directory \
            or conf['version'] != __version__ \
            or conf['dictionary_path'] != args.dictionary_path:
        logger.warning(
            'WARNING: Using old temp directory, this might not be ideal for you, use the --clean flag to ensure no '
            'weird behavior for previous versions of the temporary directory.')
        if conf['dirty']:
            logger.debug('Previous run ended in an error (maybe ctrl-c?)')
        if conf['type'] != command:
            logger.debug('Previous run was a different subcommand than {} (was {})'.format(command, conf['type']))
        if conf['corpus_directory'] != args.corpus_directory:
            logger.debug('Previous run used source directory '
                         'path {} (new run: {})'.format(conf['corpus_directory'], args.corpus_directory))
        if conf['version'] != __version__:
            logger.debug('Previous run was on {} version (new run: {})'.format(conf['version'], __version__))
        if conf['dictionary_path'] != args.dictionary_path:
            logger.debug('Previous run used dictionary path {} '
                         '(new run: {})'.format(conf['dictionary_path'], args.dictionary_path))

    os.makedirs(data_directory, exist_ok=True)
    model_directory = os.path.join(data_directory, 'acoustic_models')
    audio_dir = None
    if args.audio_directory:
        audio_dir = args.audio_directory
    try:
        corpus = AlignableCorpus(args.corpus_directory, data_directory, speaker_characters=args.speaker_characters,
                                 num_jobs=getattr(args, 'num_jobs', 3),
                                 sample_rate=align_config.feature_config.sample_frequency,
                                 debug=getattr(args, 'debug', False), logger=logger, use_mp=align_config.use_mp,
                                 punctuation=align_config.punctuation, clitic_markers=align_config.clitic_markers, audio_directory=audio_dir)
        if corpus.issues_check:
            logger.warning('Some issues parsing the corpus were detected. '
                           'Please run the validator to get more information.')
        logger.info(corpus.speaker_utterance_info())
        dictionary = Dictionary(args.dictionary_path, data_directory, word_set=corpus.word_set, logger=logger,
                                punctuation=align_config.punctuation, clitic_markers=align_config.clitic_markers,
                                compound_markers=align_config.compound_markers,
                                multilingual_ipa=align_config.multilingual_ipa,
                                strip_diacritics=align_config.strip_diacritics,
                                digraphs=align_config.digraphs)
        utt_oov_path = os.path.join(corpus.split_directory, 'utterance_oovs.txt')
        if os.path.exists(utt_oov_path):
            shutil.copy(utt_oov_path, args.output_directory)
        oov_path = os.path.join(corpus.split_directory, 'oovs_found.txt')
        if os.path.exists(oov_path):
            shutil.copy(oov_path, args.output_directory)
        a = TrainableAligner(corpus, dictionary, train_config, align_config,
                             temp_directory=data_directory, logger=logger,
                             debug=getattr(args, 'debug', False))
        a.verbose = args.verbose
        begin = time.time()
        generate_final_alignments = True
        if args.output_directory is None:
            generate_final_alignments = False
        else:
            os.makedirs(args.output_directory, exist_ok=True)

        a.train(generate_final_alignments)
        logger.debug(f'Training took {time.time() - begin} seconds')
        if args.output_model_path is not None:
            a.save(args.output_model_path, root_directory=model_directory)

        if args.output_directory is not None:
            a.export_textgrids(args.output_directory)
        logger.info('All done!')
        logger.debug(f'Done! Everything took {time.time() - all_begin} seconds')
    except Exception as _:
        conf['dirty'] = True
        raise
    finally:
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
        conf.save(conf_path)


def validate_args(args: Namespace) -> None:
    try:
        args.speaker_characters = int(args.speaker_characters)
    except ValueError:
        pass

    args.output_directory = None
    if not args.output_model_path:
        args.output_model_path = None
    output_paths = args.output_paths
    if len(output_paths) > 2:
        raise ArgumentError(f'Got more arguments for output_paths than 2: {output_paths}')
    for path in output_paths:
        if path.endswith('.zip'):
            args.output_model_path = path
        else:
            args.output_directory = path.rstrip('/').rstrip('\\')

    args.corpus_directory = args.corpus_directory.rstrip('/').rstrip('\\')
    if args.corpus_directory == args.output_directory:
        raise ArgumentError('Corpus directory and output directory cannot be the same folder.')
    if not os.path.exists(args.corpus_directory):
        raise (ArgumentError(f'Could not find the corpus directory "{args.corpus_directory}".'))
    if not os.path.isdir(args.corpus_directory):
        raise (ArgumentError(f'The specified corpus directory "{args.corpus_directory}" is not a directory.'))

    args.dictionary_path = validate_model_arg(args.dictionary_path, 'dictionary')


def run_train_corpus(args: Namespace, unknown_args: Optional[list]=None) -> None:
    validate_args(args)
    align_corpus(args, unknown_args)

