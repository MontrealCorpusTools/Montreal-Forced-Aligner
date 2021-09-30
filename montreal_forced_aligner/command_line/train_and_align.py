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
from montreal_forced_aligner.utils import get_available_dict_languages, get_dictionary_path, validate_dictionary_arg
from montreal_forced_aligner.helper import setup_logger, log_config

from montreal_forced_aligner.exceptions import ArgumentError


def align_corpus(args, unknown_args=None):
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
    assert train_config.training_configs[0].beam == 100
    train_config.use_mp = not args.disable_mp
    align_config.use_mp = not args.disable_mp
    align_config.debug = args.debug
    align_config.overwrite = args.overwrite
    align_config.cleanup_textgrids = not args.disable_textgrid_cleanup
    if unknown_args:
        train_config.update_from_args(unknown_args)
        align_config.update_from_args(unknown_args)
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
    if os.path.exists(conf_path):
        with open(conf_path, 'r') as f:
            conf = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        conf = {'dirty': False,
                'begin': time.time(),
                'version': __version__,
                'type': command,
                'corpus_directory': args.corpus_directory,
                'dictionary_path': args.dictionary_path}
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
    os.makedirs(args.output_directory, exist_ok=True)
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
        utt_oov_path = os.path.join(corpus.split_directory(), 'utterance_oovs.txt')
        if os.path.exists(utt_oov_path):
            shutil.copy(utt_oov_path, args.output_directory)
        oov_path = os.path.join(corpus.split_directory(), 'oovs_found.txt')
        if os.path.exists(oov_path):
            shutil.copy(oov_path, args.output_directory)
        a = TrainableAligner(corpus, dictionary, train_config, align_config,
                             temp_directory=data_directory, logger=logger,
                             debug=getattr(args, 'debug', False))
        a.verbose = args.verbose
        begin = time.time()
        a.train()
        logger.debug('Training took {} seconds'.format(time.time() - begin))
        if args.output_model_path is not None:
            a.save(args.output_model_path, root_directory=model_directory)
        a.export_textgrids(args.output_directory)
        logger.info('All done!')
        logger.debug('Done! Everything took {} seconds'.format(time.time() - all_begin))
    except Exception as _:
        conf['dirty'] = True
        raise
    finally:
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
        with open(conf_path, 'w') as f:
            yaml.dump(conf, f)


def validate_args(args, download_dictionaries):
    if args.corpus_directory == args.output_directory:
        raise Exception('Corpus directory and output directory cannot be the same folder.')
    if not os.path.exists(args.corpus_directory):
        raise (ArgumentError('Could not find the corpus directory {}.'.format(args.corpus_directory)))
    if not os.path.isdir(args.corpus_directory):
        raise (ArgumentError('The specified corpus directory ({}) is not a directory.'.format(args.corpus_directory)))

    args.dictionary_path = validate_dictionary_arg(args.dictionary_path, download_dictionaries)
    if not os.path.exists(args.dictionary_path):
        raise (ArgumentError('Could not find the dictionary file {}'.format(args.dictionary_path)))
    if not os.path.isfile(args.dictionary_path):
        raise (ArgumentError('The specified dictionary path ({}) is not a text file.'.format(args.dictionary_path)))


def run_train_corpus(args, unknown_args=None, download_dictionaries=None):
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
    align_corpus(args, unknown_args)

