import shutil
import os
import time
import multiprocessing as mp
import yaml

from montreal_forced_aligner import __version__
from montreal_forced_aligner.corpus.align_corpus import AlignableCorpus
from montreal_forced_aligner.dictionary import Dictionary, MultispeakerDictionary
from montreal_forced_aligner.aligner import PretrainedAligner
from montreal_forced_aligner.models import AcousticModel
from montreal_forced_aligner.config import TEMP_DIR, align_yaml_to_config, load_basic_align
from montreal_forced_aligner.utils import get_available_acoustic_languages, get_pretrained_acoustic_path, \
    get_available_dict_languages, validate_dictionary_arg
from montreal_forced_aligner.helper import setup_logger, log_config
from montreal_forced_aligner.exceptions import ArgumentError


def align_corpus(args, unknown_args=None):
    command = 'align'
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
        align_config = align_yaml_to_config(args.config_path)
    else:
        align_config = load_basic_align()
    align_config.use_mp = not args.disable_mp
    align_config.overwrite = args.overwrite
    align_config.debug = args.debug
    align_config.cleanup_textgrids = not args.disable_textgrid_cleanup
    if unknown_args:
        align_config.update_from_args(unknown_args)
    conf_path = os.path.join(data_directory, 'config.yml')
    if getattr(args, 'clean', False) and os.path.exists(data_directory):
        print('Cleaning old directory!')
        shutil.rmtree(data_directory, ignore_errors=True)
    if getattr(args, 'verbose', False):
        log_level = 'debug'
    else:
        log_level = 'info'
    logger = setup_logger(command, data_directory, console_level=log_level)
    logger.debug('ALIGN CONFIG:')
    log_config(logger, align_config)
    if os.path.exists(conf_path):
        with open(conf_path, 'r') as f:
            conf = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        conf = {'dirty': False,
                'begin': all_begin,
                'version': __version__,
                'type': command,
                'corpus_directory': args.corpus_directory,
                'dictionary_path': args.dictionary_path,
                'acoustic_model_path': args.acoustic_model_path}
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
        if conf['acoustic_model_path'] != args.acoustic_model_path:
            logger.debug('Previous run used acoustic model path {} '
                         '(new run: {})'.format(conf['acoustic_model_path'], args.acoustic_model_path))

    os.makedirs(data_directory, exist_ok=True)
    model_directory = os.path.join(data_directory, 'acoustic_models')
    os.makedirs(model_directory, exist_ok=True)
    os.makedirs(args.output_directory, exist_ok=True)
    acoustic_model = AcousticModel(args.acoustic_model_path, root_directory=model_directory)
    acoustic_model.log_details(logger)
    audio_dir = None
    if args.audio_directory:
        audio_dir = args.audio_directory
    try:
        corpus = AlignableCorpus(args.corpus_directory, data_directory,
                                 speaker_characters=args.speaker_characters,
                                 num_jobs=args.num_jobs, sample_rate=align_config.feature_config.sample_frequency,
                                 logger=logger, use_mp=align_config.use_mp, punctuation=align_config.punctuation,
                                 clitic_markers=align_config.clitic_markers, audio_directory=audio_dir)
        if corpus.issues_check:
            logger.warning('Some issues parsing the corpus were detected. '
                           'Please run the validator to get more information.')
        logger.info(corpus.speaker_utterance_info())
        if args.dictionary_path.lower().endswith('.yaml'):
            dictionary = MultispeakerDictionary(args.dictionary_path, data_directory, logger=logger,
                                                punctuation=align_config.punctuation, word_set=corpus.word_set,
                                                clitic_markers=align_config.clitic_markers,
                                                compound_markers=align_config.compound_markers,
                                                multilingual_ipa=acoustic_model.meta['multilingual_ipa'],
                                                strip_diacritics=acoustic_model.meta.get('strip_diacritics', None),
                                                digraphs=acoustic_model.meta.get('digraphs', None))
        else:
            dictionary = Dictionary(args.dictionary_path, data_directory, logger=logger,
                                    punctuation=align_config.punctuation, word_set=corpus.word_set,
                                    clitic_markers=align_config.clitic_markers,
                                    compound_markers=align_config.compound_markers,
                                    multilingual_ipa=acoustic_model.meta['multilingual_ipa'],
                                    strip_diacritics=acoustic_model.meta.get('strip_diacritics', None),
                                    digraphs=acoustic_model.meta.get('digraphs', None))
        acoustic_model.validate(dictionary)

        begin = time.time()
        a = PretrainedAligner(corpus, dictionary, acoustic_model, align_config,
                              temp_directory=data_directory,
                              debug=getattr(args, 'debug', False), logger=logger)
        logger.debug('Setup pretrained aligner in {} seconds'.format(time.time() - begin))
        a.verbose = args.verbose

        begin = time.time()
        a.align()
        logger.debug('Performed alignment in {} seconds'.format(time.time() - begin))

        begin = time.time()
        a.export_textgrids(args.output_directory)
        logger.debug('Exported TextGrids in {} seconds'.format(time.time() - begin))
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


def validate_args(args, downloaded_acoustic_models, download_dictionaries):
    if not os.path.exists(args.corpus_directory):
        raise ArgumentError('Could not find the corpus directory {}.'.format(args.corpus_directory))
    if not os.path.isdir(args.corpus_directory):
        raise ArgumentError('The specified corpus directory ({}) is not a directory.'.format(args.corpus_directory))

    if args.corpus_directory == args.output_directory:
        raise ArgumentError('Corpus directory and output directory cannot be the same folder.')

    args.dictionary_path = validate_dictionary_arg(args.dictionary_path, download_dictionaries)

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


def run_align_corpus(args, unknown_args=None, downloaded_acoustic_models=None, download_dictionaries=None):
    if downloaded_acoustic_models is None:
        downloaded_acoustic_models = get_available_acoustic_languages()
    if download_dictionaries is None:
        download_dictionaries = get_available_dict_languages()
    try:
        args.speaker_characters = int(args.speaker_characters)
    except ValueError:
        pass
    args.output_directory = args.output_directory.rstrip('/').rstrip('\\')
    args.corpus_directory = args.corpus_directory.rstrip('/').rstrip('\\')

    validate_args(args, downloaded_acoustic_models, download_dictionaries)
    align_corpus(args, unknown_args)
