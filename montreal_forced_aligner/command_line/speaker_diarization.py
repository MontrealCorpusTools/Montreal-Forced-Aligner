import shutil
import os
import time
import multiprocessing as mp
import yaml

from montreal_forced_aligner import __version__
from montreal_forced_aligner.corpus.transcribe_corpus import TranscribeCorpus
from montreal_forced_aligner.speaker_diarizer import SpeakerDiarizer
from montreal_forced_aligner.models import IvectorExtractor
from montreal_forced_aligner.config import TEMP_DIR, diarization_yaml_to_config, load_basic_diarization
from montreal_forced_aligner.utils import get_available_ivector_languages, get_pretrained_ivector_path
from montreal_forced_aligner.helper import setup_logger
from montreal_forced_aligner.exceptions import ArgumentError


def speaker_diarization(args):
    command = 'speaker_diarization'
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
    logger = setup_logger(command, data_directory)
    if args.config_path:
        diarization_config = diarization_yaml_to_config(args.config_path)
    else:
        diarization_config = load_basic_diarization()
    if getattr(args, 'clean', False) and os.path.exists(data_directory):
        logger.info('Cleaning old directory!')
        shutil.rmtree(data_directory, ignore_errors=True)
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
        if conf['type'] != 'train_ivector':
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
                        num_jobs=args.num_jobs, logger=logger)
        ivector_extractor = IvectorExtractor(args.ivector_extractor_path)

        begin = time.time()
        a = SpeakerDiarizer(corpus, ivector_extractor, diarization_config,
                              temp_directory=data_directory,
                              debug=getattr(args, 'debug', False), logger=logger)
        logger.debug('Setup speaker diarizer in {} seconds'.format(time.time() - begin))
        a.verbose = args.verbose

        begin = time.time()
        a.diarize()
        logger.debug('Performed diarization in {} seconds'.format(time.time() - begin))

        begin = time.time()
        a.export_textgrids(args.output_directory)
        logger.debug('Exported TextGrids in {} seconds'.format(time.time() - begin))
        logger.info('Done!')
        logger.debug('Done! Everything took {} seconds'.format(time.time() - all_begin))
    except Exception as _:
        conf['dirty'] = True
        raise
    finally:
        with open(conf_path, 'w') as f:
            yaml.dump(conf, f)


def validate_args(args, downloaded_ivector_extractors):
    if not os.path.exists(args.corpus_directory):
        raise ArgumentError('Could not find the corpus directory {}.'.format(args.corpus_directory))
    if not os.path.isdir(args.corpus_directory):
        raise ArgumentError('The specified corpus directory ({}) is not a directory.'.format(args.corpus_directory))

    if args.corpus_directory == args.output_directory:
        raise ArgumentError('Corpus directory and output directory cannot be the same folder.')

    if args.ivector_extractor_path.lower() in downloaded_ivector_extractors:
        args.ivector_extractor_path = get_pretrained_ivector_path(args.acoustic_model_path.lower())
    elif args.ivector_extractor_path.lower().endswith(IvectorExtractor.extension):
        if not os.path.exists(args.ivector_extractor_path):
            raise ArgumentError('The specified model path does not exist: ' + args.ivector_extractor_path)
    else:
        raise ArgumentError(
            'The language \'{}\' is not currently included in the distribution, '
            'please align via training or specify one of the following language names: {}.'.format(
                args.ivector_extractor_path.lower(), ', '.join(downloaded_ivector_extractors)))


def run_speaker_diarization(args, downloaded_ivector_extractors=None):
    if downloaded_ivector_extractors is None:
        downloaded_ivector_extractors = get_available_ivector_languages()
    args.output_directory = args.output_directory.rstrip('/').rstrip('\\')
    args.corpus_directory = args.corpus_directory.rstrip('/').rstrip('\\')

    validate_args(args, downloaded_ivector_extractors)
    speaker_diarization(args)


if __name__ == '__main__':  # pragma: no cover
    mp.freeze_support()
    from montreal_forced_aligner.command_line.mfa import speaker_diarization_parser, fix_path, unfix_path, ivector_languages

    speaker_diarization_args = speaker_diarization_parser.parse_args()
    fix_path()
    run_speaker_diarization(speaker_diarization_args, ivector_languages)
    unfix_path()
