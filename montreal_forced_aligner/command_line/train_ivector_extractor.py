import shutil
import os
import multiprocessing as mp
import yaml
import time

from montreal_forced_aligner import __version__
from montreal_forced_aligner.corpus import TranscribeCorpus
from montreal_forced_aligner.aligner import TrainableAligner
from montreal_forced_aligner.config import TEMP_DIR, train_yaml_to_config, load_basic_train_ivector
from montreal_forced_aligner.helper import setup_logger

from montreal_forced_aligner.exceptions import ArgumentError


def train_ivector(args):
    command = 'train_ivector'
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
    logger = setup_logger(command, data_directory)
    if args.config_path:
        train_config, align_config = train_yaml_to_config(args.config_path)
    else:
        train_config, align_config = load_basic_train_ivector()
    conf_path = os.path.join(data_directory, 'config.yml')
    if getattr(args, 'clean', False) and os.path.exists(data_directory):
        logger.info('Cleaning old directory!')
        shutil.rmtree(data_directory, ignore_errors=True)

    if os.path.exists(conf_path):
        with open(conf_path, 'r') as f:
            conf = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        conf = {'dirty': False,
                'begin': all_begin,
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
        if conf['type'] != 'train_ivector':
            logger.debug('Previous run was a different subcommand than {} (was {})'.format(command, conf['type']))
        if conf['corpus_directory'] != args.corpus_directory:
            logger.debug('Previous run used source directory '
                         'path {} (new run: {})'.format(conf['corpus_directory'], args.corpus_directory))
        if conf['version'] != __version__:
            logger.debug('Previous run was on {} version (new run: {})'.format(conf['version'], __version__))

    os.makedirs(data_directory, exist_ok=True)
    try:
        corpus = TranscribeCorpus(args.corpus_directory, data_directory,
                                  speaker_characters=args.speaker_characters,
                                  num_jobs=args.num_jobs,
                             debug=getattr(args, 'debug', False), logger=logger)
        dictionary = None
        a = TrainableAligner(corpus, dictionary, train_config, align_config,
                             temp_directory=data_directory, logger=logger)
        a.verbose = args.verbose
        begin = time.time()
        a.train()
        logger.debug('Training took {} seconds'.format(time.time() - begin))
        a.save(args.output_model_path)
        logger.info('All done!')
        logger.debug('Done! Everything took {} seconds'.format(time.time() - all_begin))
    except Exception as e:
        conf['dirty'] = True
        raise e
    finally:
        with open(conf_path, 'w') as f:
            yaml.dump(conf, f)


def validate_args(args):
    if args.config_path and not os.path.exists(args.config_path):
        raise (ArgumentError('Could not find the config file {}.'.format(args.config_path)))

    if not os.path.exists(args.corpus_directory):
        raise (ArgumentError('Could not find the corpus directory {}.'.format(args.corpus_directory)))
    if not os.path.isdir(args.corpus_directory):
        raise (ArgumentError('The specified corpus directory ({}) is not a directory.'.format(args.corpus_directory)))


def run_train_ivector_extractor(args):
    try:
        args.speaker_characters = int(args.speaker_characters)
    except ValueError:
        pass
    args.corpus_directory = args.corpus_directory.rstrip('/').rstrip('\\')

    validate_args(args)
    train_ivector(args)


if __name__ == '__main__':  # pragma: no cover
    mp.freeze_support()
    from montreal_forced_aligner.command_line.mfa import train_ivector_parser, fix_path, unfix_path

    ivector_args = train_ivector_parser.parse_args()

    fix_path()
    run_train_ivector_extractor(ivector_args)
    unfix_path()
