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
from montreal_forced_aligner.exceptions import ArgumentError


def speaker_diarization(args):
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
    if os.path.exists(conf_path):
        with open(conf_path, 'r') as f:
            conf = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        conf = {'dirty': False,
                'begin': time.time(),
                'version': __version__,
                'type': 'speaker_diarization',
                'corpus_directory': args.corpus_directory}
    if getattr(args, 'clean', False) \
            or conf['dirty'] or conf['type'] != 'align' \
            or conf['corpus_directory'] != args.corpus_directory \
            or conf['version'] != __version__ :
        shutil.rmtree(data_directory, ignore_errors=True)

    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(args.output_directory, exist_ok=True)
    try:
        corpus = TranscribeCorpus(args.corpus_directory, data_directory,
                        speaker_characters=args.speaker_characters,
                        num_jobs=args.num_jobs)
        print(corpus.speaker_utterance_info())
        ivector_extractor = IvectorExtractor(args.ivector_extractor_path)

        begin = time.time()
        if args.config_path:
            diarization_config = diarization_yaml_to_config(args.config_path)
        else:
            diarization_config = load_basic_diarization()
        a = SpeakerDiarizer(corpus, ivector_extractor, diarization_config,
                              temp_directory=data_directory,
                              debug=getattr(args, 'debug', False))
        if args.debug:
            print('Setup pretrained aligner in {} seconds'.format(time.time() - begin))
        a.verbose = args.verbose

        begin = time.time()
        a.diarize()
        if args.debug:
            print('Performed alignment in {} seconds'.format(time.time() - begin))

        begin = time.time()
        a.export_textgrids(args.output_directory)
        if args.debug:
            print('Exported TextGrids in {} seconds'.format(time.time() - begin))
        print('Done! Everything took {} seconds'.format(time.time() - all_begin))
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
    try:
        args.speaker_characters = int(args.speaker_characters)
    except ValueError:
        pass
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
