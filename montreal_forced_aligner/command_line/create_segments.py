import shutil
import os
import time
import multiprocessing as mp
import yaml

from montreal_forced_aligner import __version__
from montreal_forced_aligner.corpus.transcribe_corpus import TranscribeCorpus
from montreal_forced_aligner.segmenter import Segmenter
from montreal_forced_aligner.config import TEMP_DIR, segmentation_yaml_to_config, load_basic_segmentation
from montreal_forced_aligner.helper import setup_logger, log_config
from montreal_forced_aligner.exceptions import ArgumentError


def create_segments(args, unknown_args=None):
    command = 'create_segments'
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
        segmentation_config = segmentation_yaml_to_config(args.config_path)
    else:
        segmentation_config = load_basic_segmentation()
    segmentation_config.use_mp = not args.disable_mp
    if unknown_args:
        segmentation_config.update_from_args(unknown_args)
    if getattr(args, 'clean', False) and os.path.exists(data_directory):
        print('Cleaning old directory!')
        shutil.rmtree(data_directory, ignore_errors=True)
    if getattr(args, 'verbose', False):
        log_level = 'debug'
    else:
        log_level = 'info'
    logger = setup_logger(command, data_directory, console_level=log_level)
    log_config(logger, segmentation_config)
    if os.path.exists(conf_path):
        with open(conf_path, 'r') as f:
            conf = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        conf = {'dirty': False,
                'begin': time.time(),
                'version': __version__,
                'type': command,
                'corpus_directory': args.corpus_directory}
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

    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(args.output_directory, exist_ok=True)
    try:
        corpus = TranscribeCorpus(args.corpus_directory, data_directory,
                                  sample_rate=segmentation_config.feature_config.sample_frequency,
                        num_jobs=args.num_jobs, logger=logger, use_mp=segmentation_config.use_mp, no_speakers=True,
                                  ignore_transcriptions=True)

        begin = time.time()
        a = Segmenter(corpus, segmentation_config,
                              temp_directory=data_directory,
                              debug=getattr(args, 'debug', False), logger=logger)
        logger.debug('Setup segmenter in {} seconds'.format(time.time() - begin))
        a.verbose = args.verbose

        begin = time.time()
        a.segment()
        logger.debug('Performed segmentation in {} seconds'.format(time.time() - begin))

        begin = time.time()
        a.export_segments(args.output_directory)
        logger.debug('Exported segmentation in {} seconds'.format(time.time() - begin))
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


def validate_args(args):
    if not os.path.exists(args.corpus_directory):
        raise ArgumentError('Could not find the corpus directory {}.'.format(args.corpus_directory))
    if not os.path.isdir(args.corpus_directory):
        raise ArgumentError('The specified corpus directory ({}) is not a directory.'.format(args.corpus_directory))

    if args.corpus_directory == args.output_directory:
        raise ArgumentError('Corpus directory and output directory cannot be the same folder.')


def run_create_segments(args, unknown=None):
    args.output_directory = args.output_directory.rstrip('/').rstrip('\\')
    args.corpus_directory = args.corpus_directory.rstrip('/').rstrip('\\')

    validate_args(args)
    create_segments(args, unknown)