import shutil
import os
import time
import multiprocessing as mp
import yaml

from montreal_forced_aligner import __version__
from montreal_forced_aligner.corpus import AlignableCorpus, TranscribeCorpus
from montreal_forced_aligner.dictionary import Dictionary, MultispeakerDictionary
from montreal_forced_aligner.transcriber import Transcriber
from montreal_forced_aligner.models import AcousticModel, LanguageModel, FORMAT
from montreal_forced_aligner.helper import setup_logger, log_config
from montreal_forced_aligner.config import TEMP_DIR, transcribe_yaml_to_config, load_basic_transcribe, save_config
from montreal_forced_aligner.utils import get_available_acoustic_languages, get_pretrained_acoustic_path, \
    get_available_lm_languages, get_pretrained_language_model_path, \
    get_available_dict_languages, validate_dictionary_arg
from montreal_forced_aligner.exceptions import ArgumentError


def transcribe_corpus(args, unknown_args):
    command = 'transcribe'
    all_begin = time.time()
    if not args.temp_directory:
        temp_dir = TEMP_DIR
    else:
        temp_dir = os.path.expanduser(args.temp_directory)
    corpus_name = os.path.basename(args.corpus_directory)
    if corpus_name == '':
        args.corpus_directory = os.path.dirname(args.corpus_directory)
        corpus_name = os.path.basename(args.corpus_directory)
    if args.config_path:
        transcribe_config = transcribe_yaml_to_config(args.config_path)
    else:
        transcribe_config = load_basic_transcribe()
    transcribe_config.use_mp = not args.disable_mp
    transcribe_config.overwrite = args.overwrite
    if unknown_args:
        transcribe_config.update_from_args(unknown_args)
    data_directory = os.path.join(temp_dir, corpus_name)
    if getattr(args, 'clean', False) and os.path.exists(data_directory):
        print('Cleaning old directory!')
        shutil.rmtree(data_directory, ignore_errors=True)
    if getattr(args, 'verbose', False):
        log_level = 'debug'
    else:
        log_level = 'info'
    logger = setup_logger(command, data_directory, console_level=log_level)
    logger.debug('TRANSCRIBE CONFIG:')
    log_config(logger, transcribe_config)
    os.makedirs(data_directory, exist_ok=True)
    model_directory = os.path.join(data_directory, 'acoustic_models')
    os.makedirs(args.output_directory, exist_ok=True)
    os.makedirs(model_directory, exist_ok=True)
    conf_path = os.path.join(data_directory, 'config.yml')
    if os.path.exists(conf_path):
        with open(conf_path, 'r') as f:
            conf = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        conf = {'dirty': False,
                'begin': time.time(),
                'version': __version__,
                'type': 'transcribe',
                'corpus_directory': args.corpus_directory,
                'dictionary_path': args.dictionary_path,
                'acoustic_model_path': args.acoustic_model_path,
                'language_model_path': args.language_model_path,
                }
    if conf['dirty'] or conf['type'] != command \
            or conf['corpus_directory'] != args.corpus_directory \
            or conf['version'] != __version__ \
            or conf['dictionary_path'] != args.dictionary_path \
            or conf['language_model_path'] != args.language_model_path \
            or conf['acoustic_model_path'] != args.acoustic_model_path:
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
        if conf['language_model_path'] != args.language_model_path:
            logger.debug('Previous run used language model path {} '
                         '(new run: {})'.format(conf['language_model_path'], args.language_model_path))
    audio_dir = None
    if args.audio_directory:
        audio_dir = args.audio_directory
    try:
        if args.evaluate:
            corpus = AlignableCorpus(args.corpus_directory, data_directory,
                                     speaker_characters=args.speaker_characters,
                                     sample_rate=transcribe_config.feature_config.sample_frequency,
                                     num_jobs=args.num_jobs, use_mp=transcribe_config.use_mp, audio_directory=audio_dir)
        else:
            corpus = TranscribeCorpus(args.corpus_directory, data_directory,
                                      speaker_characters=args.speaker_characters,
                                      sample_rate=transcribe_config.feature_config.sample_frequency,
                                      num_jobs=args.num_jobs, use_mp=transcribe_config.use_mp,
                                      no_speakers=transcribe_config.no_speakers, audio_directory=audio_dir)
        acoustic_model = AcousticModel(args.acoustic_model_path, root_directory=model_directory)
        acoustic_model.log_details(logger)
        if args.language_model_path.endswith('.arpa'):
            alternative_name = os.path.splitext(args.language_model_path)[0] + '.zip'
            logger.warning(f"Using a plain .arpa model requires generating pruned versions of it to decode in a reasonable "
                           f"amount of time.  If you'd like to generate a reusable language model, consider running "
                           f"`mfa train_lm {args.language_model_path} {alternative_name}`.")
        language_model = LanguageModel(args.language_model_path, root_directory=data_directory)
        if args.dictionary_path.lower().endswith('.yaml'):
            dictionary = MultispeakerDictionary(args.dictionary_path, data_directory, logger=logger,
                                                punctuation=transcribe_config.punctuation,
                                                clitic_markers=transcribe_config.clitic_markers,
                                                compound_markers=transcribe_config.compound_markers,
                                multilingual_ipa=acoustic_model.meta['multilingual_ipa'])
        else:
            dictionary = Dictionary(args.dictionary_path, data_directory, logger=logger,
                                                punctuation=transcribe_config.punctuation,
                                                clitic_markers=transcribe_config.clitic_markers,
                                                compound_markers=transcribe_config.compound_markers,
                                multilingual_ipa=acoustic_model.meta['multilingual_ipa'])
        acoustic_model.validate(dictionary)
        begin = time.time()
        t = Transcriber(corpus, dictionary, acoustic_model, language_model, transcribe_config,
                        temp_directory=data_directory,
                        debug=getattr(args, 'debug', False), evaluation_mode=args.evaluate, logger=logger)
        logger.debug('Setup pretrained aligner in {} seconds'.format(time.time() - begin))

        begin = time.time()
        t.transcribe()
        logger.debug('Performed transcribing in {} seconds'.format(time.time() - begin))
        if args.evaluate:
            t.evaluate(args.output_directory)
            best_config_path = os.path.join(args.output_directory, 'best_transcribe_config.yaml')
            save_config(t.transcribe_config, best_config_path)
            t.export_transcriptions(args.output_directory)
        else:
            begin = time.time()
            t.export_transcriptions(args.output_directory)
            logger.debug('Exported transcriptions in {} seconds'.format(time.time() - begin))
        logger.info('Done! Everything took {} seconds'.format(time.time() - all_begin))
    except Exception as _:
        conf['dirty'] = True
        raise
    finally:
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
        if os.path.exists(data_directory):
            with open(conf_path, 'w', encoding='utf8') as f:
                yaml.dump(conf, f)


def validate_args(args, downloaded_acoustic_models, download_dictionaries,  downloaded_language_models):
    validate_dictionary_arg(args.dictionary_path, download_dictionaries)
    if not os.path.exists(args.corpus_directory):
        raise ArgumentError('Could not find the corpus directory {}.'.format(args.corpus_directory))
    if not os.path.isdir(args.corpus_directory):
        raise ArgumentError('The specified corpus directory ({}) is not a directory.'.format(args.corpus_directory))

    if args.corpus_directory == args.output_directory:
        raise ArgumentError('Corpus directory and output directory cannot be the same folder.')

    if args.acoustic_model_path.lower() in downloaded_acoustic_models:
        args.acoustic_model_path = get_pretrained_acoustic_path(args.acoustic_model_path.lower())
    elif args.acoustic_model_path.lower().endswith(AcousticModel.extension):
        if not os.path.exists(args.acoustic_model_path):
            raise ArgumentError('The specified model path does not exist: ' + args.acoustic_model_path)
    else:
        raise ArgumentError(
            'The acoustic model \'{}\' is not currently downloaded, '
            'please download a pretrained model, align via training or specify one of the following language names: {}.'.format(
                args.acoustic_model_path.lower(), ', '.join(downloaded_acoustic_models)))

    if args.language_model_path.lower() in downloaded_language_models:
        args.language_model_path = get_pretrained_language_model_path(args.language_model_path.lower())
    elif args.language_model_path.lower().endswith(LanguageModel.extension) or \
            args.language_model_path.lower().endswith(FORMAT):
        if not os.path.exists(args.language_model_path):
            raise ArgumentError('The specified model path does not exist: ' + args.language_model_path)
    else:
        raise ArgumentError(
            'The language model \'{}\' is not currently downloaded, '
            'please download, train a new model, or specify one of the following language names: {}.'.format(
                args.language_model_path.lower(), ', '.join(downloaded_language_models)))


def run_transcribe_corpus(args, unknown=None, downloaded_acoustic_models=None, download_dictionaries=None,
                          downloaded_language_models=None):
    if downloaded_acoustic_models is None:
        downloaded_acoustic_models = get_available_acoustic_languages()
    if download_dictionaries is None:
        download_dictionaries = get_available_dict_languages()
    if downloaded_language_models is None:
        downloaded_language_models = get_available_lm_languages()
    try:
        args.speaker_characters = int(args.speaker_characters)
    except ValueError:
        pass
    args.output_directory = args.output_directory.rstrip('/').rstrip('\\')
    args.corpus_directory = args.corpus_directory.rstrip('/').rstrip('\\')

    validate_args(args, downloaded_acoustic_models, download_dictionaries, downloaded_language_models)
    transcribe_corpus(args, unknown)

