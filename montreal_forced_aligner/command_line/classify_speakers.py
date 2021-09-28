import shutil
import os
import time
import multiprocessing as mp
import yaml

from montreal_forced_aligner import __version__
from montreal_forced_aligner.corpus.transcribe_corpus import TranscribeCorpus
from montreal_forced_aligner.speaker_classifier import SpeakerClassifier
from montreal_forced_aligner.models import IvectorExtractor
from montreal_forced_aligner.config import TEMP_DIR, classification_yaml_to_config, load_basic_classification
from montreal_forced_aligner.utils import get_available_ivector_languages, get_pretrained_ivector_path
from montreal_forced_aligner.helper import setup_logger
from montreal_forced_aligner.exceptions import ArgumentError


def classify_speakers(args, unknown_args=None):
    command = 'classify_speakers'
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
    conf_path = os.path.join(data_directory, 'config.yml')
    if args.config_path:
        classification_config = classification_yaml_to_config(args.config_path)
    else:
        classification_config = load_basic_classification()
    classification_config.use_mp = not args.disable_mp
    if unknown_args:
        classification_config.update_from_args(unknown_args)
    classification_config.use_mp = not args.disable_mp
    if getattr(args, 'clean', False) and os.path.exists(data_directory):
        print('Cleaning old directory!')
        shutil.rmtree(data_directory, ignore_errors=True)
    if getattr(args, 'verbose', False):
        log_level = 'debug'
    else:
        log_level = 'info'
    logger = setup_logger(command, data_directory, console_level=log_level)
    if os.path.exists(conf_path):
        with open(conf_path, 'r') as f:
            conf = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        conf = {'dirty': False,
                'begin': time.time(),
                'version': __version__,
                'type': command,
                'corpus_directory': args.corpus_directory,
                'ivector_extractor_path': args.ivector_extractor_path}
    if conf['dirty'] or conf['type'] != command \
            or conf['corpus_directory'] != args.corpus_directory \
            or conf['version'] != __version__:
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
        if conf['ivector_extractor_path'] != args.ivector_extractor_path:
            logger.debug('Previous run used ivector extractor path {} '
                         '(new run: {})'.format(conf['ivector_extractor_path'], args.ivector_extractor_path))

    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(args.output_directory, exist_ok=True)
    try:
        ivector_extractor = IvectorExtractor(args.ivector_extractor_path, root_directory=data_directory)
        corpus = TranscribeCorpus(args.corpus_directory, data_directory,
                                  sample_rate=ivector_extractor.feature_config.sample_frequency,
                        num_jobs=args.num_jobs, logger=logger, use_mp=classification_config.use_mp)

        begin = time.time()
        a = SpeakerClassifier(corpus, ivector_extractor, classification_config,
                              temp_directory=data_directory,
                              debug=getattr(args, 'debug', False), logger=logger, num_speakers=args.num_speakers,
                              cluster=args.cluster)
        logger.debug('Setup speaker classifier in {} seconds'.format(time.time() - begin))
        a.verbose = args.verbose

        begin = time.time()
        a.classify()
        logger.debug('Performed classification in {} seconds'.format(time.time() - begin))

        begin = time.time()
        a.export_classification(args.output_directory)
        logger.debug('Exported classification in {} seconds'.format(time.time() - begin))
        logger.info('Done!')
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


def validate_args(args, downloaded_ivector_extractors):
    if args.cluster and not args.num_speakers:
        raise ArgumentError('If using clustering, num_speakers must be specified')
    if not os.path.exists(args.corpus_directory):
        raise ArgumentError('Could not find the corpus directory {}.'.format(args.corpus_directory))
    if not os.path.isdir(args.corpus_directory):
        raise ArgumentError('The specified corpus directory ({}) is not a directory.'.format(args.corpus_directory))

    if args.corpus_directory == args.output_directory:
        raise ArgumentError('Corpus directory and output directory cannot be the same folder.')

    if args.ivector_extractor_path.lower() in downloaded_ivector_extractors:
        args.ivector_extractor_path = get_pretrained_ivector_path(args.ivector_extractor_path.lower())
    elif args.ivector_extractor_path.lower().endswith(IvectorExtractor.extension):
        if not os.path.exists(args.ivector_extractor_path):
            raise ArgumentError('The specified model path does not exist: ' + args.ivector_extractor_path)
    else:
        raise ArgumentError(
            'The language \'{}\' is not currently included in the distribution, '
            'please align via training or specify one of the following language names: {}.'.format(
                args.ivector_extractor_path.lower(), ', '.join(downloaded_ivector_extractors)))


def run_classify_speakers(args, unknown=None, downloaded_ivector_extractors=None):
    if downloaded_ivector_extractors is None:
        downloaded_ivector_extractors = get_available_ivector_languages()
    args.output_directory = args.output_directory.rstrip('/').rstrip('\\')
    args.corpus_directory = args.corpus_directory.rstrip('/').rstrip('\\')

    validate_args(args, downloaded_ivector_extractors)
    classify_speakers(args)

