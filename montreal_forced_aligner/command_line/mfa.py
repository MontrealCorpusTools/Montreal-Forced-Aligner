import sys
import os
import argparse
import multiprocessing as mp

from montreal_forced_aligner import __version__

from montreal_forced_aligner.utils import get_available_acoustic_languages, get_available_g2p_languages, \
    get_available_dict_languages, get_available_lm_languages, get_available_ivector_languages
from montreal_forced_aligner.command_line.align import run_align_corpus
from montreal_forced_aligner.command_line.adapt import run_adapt_model
from montreal_forced_aligner.command_line.train_and_align import run_train_corpus
from montreal_forced_aligner.command_line.g2p import run_g2p
from montreal_forced_aligner.command_line.train_g2p import run_train_g2p
from montreal_forced_aligner.command_line.validate import run_validate_corpus
from montreal_forced_aligner.command_line.download import run_download
from montreal_forced_aligner.command_line.train_lm import run_train_lm
from montreal_forced_aligner.command_line.thirdparty import run_thirdparty
from montreal_forced_aligner.command_line.train_ivector_extractor import run_train_ivector_extractor
from montreal_forced_aligner.command_line.classify_speakers import run_classify_speakers
from montreal_forced_aligner.command_line.transcribe import run_transcribe_corpus
from montreal_forced_aligner.command_line.train_dictionary import run_train_dictionary
from montreal_forced_aligner.command_line.create_segments import run_create_segments
from montreal_forced_aligner.exceptions import MFAError


def fix_path():
    from montreal_forced_aligner.config import TEMP_DIR
    thirdparty_dir = os.path.join(TEMP_DIR, 'thirdparty', 'bin')
    old_path = os.environ.get('PATH', '')
    if sys.platform == 'win32':
        os.environ['PATH'] = thirdparty_dir + ';' + old_path
    else:
        os.environ['PATH'] = thirdparty_dir + ':' + old_path
        os.environ['LD_LIBRARY_PATH'] = thirdparty_dir + ':' + os.environ.get('LD_LIBRARY_PATH', '')


def unfix_path():
    if sys.platform == 'win32':
        sep = ';'
        os.environ['PATH'] = sep.join(os.environ['PATH'].split(sep)[1:])
    else:
        sep = ':'
        os.environ['PATH'] = sep.join(os.environ['PATH'].split(sep)[1:])
        os.environ['LD_LIBRARY_PATH'] = sep.join(os.environ['PATH'].split(sep)[1:])


acoustic_languages = get_available_acoustic_languages()
ivector_languages = get_available_ivector_languages()
lm_languages = get_available_lm_languages()
g2p_languages = get_available_g2p_languages()
dict_languages = get_available_dict_languages()

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(dest="subcommand")
subparsers.required = True

version_parser = subparsers.add_parser('version')

align_parser = subparsers.add_parser('align')
align_parser.add_argument('corpus_directory', help='Full path to the directory to align')
align_parser.add_argument('dictionary_path', help='Full path to the pronunciation dictionary to use')
align_parser.add_argument('acoustic_model_path',
                          help='Full path to the archive containing pre-trained model or language ({})'.format(
                              ', '.join(acoustic_languages)))
align_parser.add_argument('output_directory',
                          help="Full path to output directory, will be created if it doesn't exist")
align_parser.add_argument('--config_path', type=str, default='',
                          help='Path to config file to use for alignment')
align_parser.add_argument('-s', '--speaker_characters', type=str, default='0',
                          help='Number of characters of file names to use for determining speaker, '
                               'default is to use directory names')
align_parser.add_argument('-t', '--temp_directory', type=str, default='',
                          help='Temporary directory root to use for aligning, default is ~/Documents/MFA')
align_parser.add_argument('-j', '--num_jobs', type=int, default=3,
                          help='Number of cores to use while aligning')
align_parser.add_argument('-v', '--verbose', help="Print more information during alignment", action='store_true')
align_parser.add_argument('-c', '--clean', help="Remove files from previous runs", action='store_true')
align_parser.add_argument('-d', '--debug', help="Output debug messages about alignment", action='store_true')

adapt_parser = subparsers.add_parser('adapt')
adapt_parser.add_argument('corpus_directory', help='Full path to the directory to align')
adapt_parser.add_argument('dictionary_path', help='Full path to the pronunciation dictionary to use')
adapt_parser.add_argument('acoustic_model_path',
                          help='Full path to the archive containing pre-trained model or language ({})'.format(
                              ', '.join(acoustic_languages)))
adapt_parser.add_argument('output_model_path',
                          help="Full path to save adapted_model")
adapt_parser.add_argument('--config_path', type=str, default='',
                          help='Path to config file to use for alignment')
adapt_parser.add_argument('-s', '--speaker_characters', type=str, default='0',
                          help='Number of characters of file names to use for determining speaker, '
                               'default is to use directory names')
adapt_parser.add_argument('-t', '--temp_directory', type=str, default='',
                          help='Temporary directory root to use for aligning, default is ~/Documents/MFA')
adapt_parser.add_argument('-j', '--num_jobs', type=int, default=3,
                          help='Number of cores to use while aligning')
adapt_parser.add_argument('-v', '--verbose', help="Print more information during alignment", action='store_true')
adapt_parser.add_argument('-c', '--clean', help="Remove files from previous runs", action='store_true')
adapt_parser.add_argument('-d', '--debug', help="Output debug messages about alignment", action='store_true')

train_parser = subparsers.add_parser('train')
train_parser.add_argument('corpus_directory', help='Full path to the source directory to align')
train_parser.add_argument('dictionary_path', help='Full path to the pronunciation dictionary to use',
                          default='')
train_parser.add_argument('output_directory',
                          help="Full path to output directory, will be created if it doesn't exist")
train_parser.add_argument('--config_path', type=str, default='',
                          help='Path to config file to use for training and alignment')
train_parser.add_argument('-o', '--output_model_path', type=str, default='',
                          help='Full path to save resulting acoustic and dictionary model')
train_parser.add_argument('-s', '--speaker_characters', type=str, default='0',
                          help='Number of characters of filenames to use for determining speaker, '
                               'default is to use directory names')
train_parser.add_argument('-t', '--temp_directory', type=str, default='',
                          help='Temporary directory root to use for aligning, default is ~/Documents/MFA')
train_parser.add_argument('-j', '--num_jobs', type=int, default=3,
                          help='Number of cores to use while aligning')
train_parser.add_argument('-v', '--verbose', help="Output debug messages about alignment", action='store_true')
train_parser.add_argument('-c', '--clean', help="Remove files from previous runs", action='store_true')
train_parser.add_argument('-d', '--debug', help="Debug the aligner", action='store_true')

validate_parser = subparsers.add_parser('validate')
validate_parser.add_argument('corpus_directory', help='Full path to the source directory to align')
validate_parser.add_argument('dictionary_path', help='Full path to the pronunciation dictionary to use',
                             default='')
validate_parser.add_argument('acoustic_model_path',
                             help='Full path to the archive containing pre-trained model or language ({})'.format(
                                 ', '.join(acoustic_languages)), nargs='?',
                             default='')
validate_parser.add_argument('-s', '--speaker_characters', type=str, default='0',
                             help='Number of characters of file names to use for determining speaker, '
                                  'default is to use directory names')
validate_parser.add_argument('-t', '--temp_directory', type=str, default='',
                             help='Temporary directory root to use for aligning, default is ~/Documents/MFA')
validate_parser.add_argument('--test_transcriptions', help="Test accuracy of transcriptions", action='store_true')
validate_parser.add_argument('--ignore_acoustics',
                             help="Skip acoustic feature generation and associated validation",
                             action='store_true')
validate_parser.add_argument('-j', '--num_jobs', type=int, default=3,
                             help='Number of cores to use while aligning')
validate_parser.add_argument('-d', '--debug', help="Debug the aligner", action='store_true')
validate_parser.add_argument('--disable_mp', help="Disable any multiprocessing during validation", action='store_true')

g2p_model_help_message = '''Full path to the archive containing pre-trained model or language ({})
If not specified, then orthographic transcription is split into pronunciations.'''.format(', '.join(g2p_languages))
g2p_parser = subparsers.add_parser('g2p')
g2p_parser.add_argument("g2p_model_path", help=g2p_model_help_message, nargs='?')

g2p_parser.add_argument("input_path",
                        help="Corpus to base word list on or a text file of words to generate pronunciations")
g2p_parser.add_argument("output_path", help="Path to save output dictionary")
g2p_parser.add_argument('-t', '--temp_directory', type=str, default='',
                        help='Temporary directory root to use for dictionary generation, default is ~/Documents/MFA')
g2p_parser.add_argument('--include_bracketed', help="Included words enclosed by brackets, i.e. [...], (...), <...>",
                        action='store_true')
g2p_parser.add_argument('-j', '--num_jobs', type=int, default=3,
                        help='Number of cores to use while generating pronunciations')
g2p_parser.add_argument('--config_path', type=str, default='',
                          help='Path to config file to use for G2P')
g2p_parser.add_argument('-c', '--clean', help="Remove files from previous runs", action='store_true')
g2p_parser.add_argument('--disable_mp', help="Disable multiprocessing (not recommended)", action='store_true')

train_g2p_parser = subparsers.add_parser('train_g2p')
train_g2p_parser.add_argument("dictionary_path", help="Location of existing dictionary")

train_g2p_parser.add_argument("output_model_path", help="Desired location of generated model")
train_g2p_parser.add_argument('-j', '--num_jobs', type=int, default=3,
                              help='Number of cores to use while training')
train_g2p_parser.add_argument('-t', '--temp_directory', type=str, default='',
                              help='Temporary directory root to use for G2P training, default is ~/Documents/MFA')
train_g2p_parser.add_argument('--config_path', type=str, default='',
                          help='Path to config file to use for G2P')
train_g2p_parser.add_argument('-v', "--verbose", action='store_true',
                              help="Output messages on the command line")
train_g2p_parser.add_argument("--validate", action='store_true',
                              help="Perform an analysis of accuracy training on "
                                   "most of the data and validating on an unseen subset")
train_g2p_parser.add_argument('-c', '--clean', help="Remove files from previous runs", action='store_true')
train_g2p_parser.add_argument('--disable_mp', help="Disable multiprocessing (not recommended)", action='store_true')

download_parser = subparsers.add_parser('download')
download_parser.add_argument("model_type",
                             help="Type of model to download, one of 'acoustic', 'g2p', or 'dictionary'")
download_parser.add_argument("language", help="Name of language code to download, if not specified, "
                                              "will list all available languages", nargs='?')

train_lm_parser = subparsers.add_parser('train_lm')
train_lm_parser.add_argument('source_path', help='Full path to the source directory to train from, alternatively '
                                                 'an ARPA format language model to convert for MFA use')
train_lm_parser.add_argument('output_model_path', type=str,
                             help='Full path to save resulting language model')
train_lm_parser.add_argument('-m', '--model_path', type=str,
                             help='Full path to existing language model to merge probabilities')
train_lm_parser.add_argument('-w', '--model_weight', type=float, default=1.0,
                             help='Weight factor for supplemental language model, defaults to 1.0')
train_lm_parser.add_argument('-d', '--dictionary_path', help='Full path to the pronunciation dictionary to use',
                             default='')
train_lm_parser.add_argument('-t', '--temp_directory', type=str, default='',
                             help='Temporary directory root to use for training, default is ~/Documents/MFA')
train_lm_parser.add_argument('-j', '--num_jobs', type=int, default=3,
                             help='Number of cores to use while aligning')
train_lm_parser.add_argument('-c', '--clean', help="Remove files from previous runs", action='store_true')
train_lm_parser.add_argument('--config_path', type=str, default='',
                             help='Path to config file to use for training and alignment')
train_lm_parser.add_argument('--debug', help="Output debug messages about training", action='store_true')

train_dictionary_parser = subparsers.add_parser('train_dictionary')
train_dictionary_parser.add_argument('corpus_directory', help='Full path to the directory to align')
train_dictionary_parser.add_argument('dictionary_path', help='Full path to the pronunciation dictionary to use')
train_dictionary_parser.add_argument('acoustic_model_path',
                                     help='Full path to the archive containing pre-trained model or language ({})'.format(
                                         ', '.join(acoustic_languages)))
train_dictionary_parser.add_argument('output_directory',
                                     help="Full path to output directory, will be created if it doesn't exist")
train_dictionary_parser.add_argument('--config_path', type=str, default='',
                                     help='Path to config file to use for alignment')
train_dictionary_parser.add_argument('-s', '--speaker_characters', type=str, default='0',
                                     help='Number of characters of file names to use for determining speaker, '
                                          'default is to use directory names')
train_dictionary_parser.add_argument('-t', '--temp_directory', type=str, default='',
                                     help='Temporary directory root to use for aligning, default is ~/Documents/MFA')
train_dictionary_parser.add_argument('-j', '--num_jobs', type=int, default=3,
                                     help='Number of cores to use while aligning')
train_dictionary_parser.add_argument('-v', '--verbose', help="Print more information during alignment",
                                     action='store_true')
train_dictionary_parser.add_argument('-c', '--clean', help="Remove files from previous runs", action='store_true')
train_dictionary_parser.add_argument('-d', '--debug', help="Output debug messages about alignment", action='store_true')

train_ivector_parser = subparsers.add_parser('train_ivector')
train_ivector_parser.add_argument('corpus_directory', help='Full path to the source directory to '
                                                           'train the ivector extractor')
train_ivector_parser.add_argument('dictionary_path', help='Full path to the pronunciation dictionary to use')
train_ivector_parser.add_argument('acoustic_model_path', type=str, default='',
                                  help='Full path to acoustic model for alignment')
train_ivector_parser.add_argument('output_model_path', type=str, default='',
                                  help='Full path to save resulting ivector extractor')
train_ivector_parser.add_argument('-s', '--speaker_characters', type=str, default='0',
                                  help='Number of characters of filenames to use for determining speaker, '
                                       'default is to use directory names')
train_ivector_parser.add_argument('-t', '--temp_directory', type=str, default='',
                                  help='Temporary directory root to use for aligning, default is ~/Documents/MFA')
train_ivector_parser.add_argument('-j', '--num_jobs', type=int, default=3,
                                  help='Number of cores to use while aligning')
train_ivector_parser.add_argument('-v', '--verbose', help="Output debug messages about alignment", action='store_true')
train_ivector_parser.add_argument('-c', '--clean', help="Remove files from previous runs", action='store_true')
train_ivector_parser.add_argument('-d', '--debug', help="Debug the aligner", action='store_true')
train_ivector_parser.add_argument('--config_path', type=str, default='',
                                  help='Path to config file to use for training')

classify_speakers_parser = subparsers.add_parser('classify_speakers')
classify_speakers_parser.add_argument('corpus_directory', help='Full path to the source directory to '
                                                               'run speaker classification')
classify_speakers_parser.add_argument('ivector_extractor_path', type=str, default='',
                                      help='Full path to ivector extractor model')
classify_speakers_parser.add_argument('output_directory',
                                      help="Full path to output directory, will be created if it doesn't exist")

classify_speakers_parser.add_argument('-s', '--num_speakers', type=int, default=0,
                                      help='Number of speakers if known')
classify_speakers_parser.add_argument('--cluster', help="Using clustering instead of classification", action='store_true')
classify_speakers_parser.add_argument('-t', '--temp_directory', type=str, default='',
                                      help='Temporary directory root to use for aligning, default is ~/Documents/MFA')
classify_speakers_parser.add_argument('-j', '--num_jobs', type=int, default=3,
                                      help='Number of cores to use while performing speaker classification')
classify_speakers_parser.add_argument('-v', '--verbose', help="Output debug messages about speaker classification",
                                      action='store_true')
classify_speakers_parser.add_argument('-c', '--clean', help="Remove files from previous runs", action='store_true')
classify_speakers_parser.add_argument('-d', '--debug', help="Debug the aligner", action='store_true')
classify_speakers_parser.add_argument('--disable_mp', help="Disable multiprocessing (not recommended)",
                                      action='store_true')
classify_speakers_parser.add_argument('--config_path', type=str, default='',
                                      help='Path to config file to use for ivector extraction')

create_segments_parser = subparsers.add_parser('create_segments')
create_segments_parser.add_argument('corpus_directory', help='Full path to the source directory to '
                                                               'run VAD segmentation')
create_segments_parser.add_argument('output_directory',
                                      help="Full path to output directory, will be created if it doesn't exist")

create_segments_parser.add_argument('-t', '--temp_directory', type=str, default='',
                                      help='Temporary directory root to use for segmentation, default is '
                                           '~/Documents/MFA')
create_segments_parser.add_argument('-j', '--num_jobs', type=int, default=3,
                                      help='Number of cores to use while creating segments')
create_segments_parser.add_argument('-v', '--verbose', help="Output debug messages about segmentation",
                                      action='store_true')
create_segments_parser.add_argument('-c', '--clean', help="Remove files from previous runs", action='store_true')
create_segments_parser.add_argument('-d', '--debug', help="Debug the aligner", action='store_true')
create_segments_parser.add_argument('--config_path', type=str, default='',
                                      help='Path to config file to use for segmentation')

transcribe_parser = subparsers.add_parser('transcribe')
transcribe_parser.add_argument('corpus_directory', help='Full path to the directory to transcribe')
transcribe_parser.add_argument('dictionary_path', help='Full path to the pronunciation dictionary to use')
transcribe_parser.add_argument('acoustic_model_path',
                               help='Full path to the archive containing pre-trained model or language ({})'.format(
                                   ', '.join(acoustic_languages)))
transcribe_parser.add_argument('language_model_path',
                               help='Full path to the archive containing pre-trained model or language ({})'.format(
                                   ', '.join(lm_languages)))
transcribe_parser.add_argument('output_directory',
                               help="Full path to output directory, will be created if it doesn't exist")
transcribe_parser.add_argument('--config_path', type=str, default='',
                               help='Path to config file to use for transcription')
transcribe_parser.add_argument('-s', '--speaker_characters', type=str, default='0',
                               help='Number of characters of file names to use for determining speaker, '
                                    'default is to use directory names')
transcribe_parser.add_argument('-t', '--temp_directory', type=str, default='',
                               help='Temporary directory root to use for transcribing, default is ~/Documents/MFA')
transcribe_parser.add_argument('-a', '--audio_directory', type=str, default='',
                               help='Audio directory root to use for finding audio files')
transcribe_parser.add_argument('-j', '--num_jobs', type=int, default=3,
                               help='Number of cores to use while aligning')
transcribe_parser.add_argument('-v', '--verbose', help="Print more information during alignment", action='store_true')
transcribe_parser.add_argument('-e', '--evaluate', help="Evaluate the transcription "
                                                        "against golden texts", action='store_true')
transcribe_parser.add_argument('-c', '--clean', help="Remove files from previous runs", action='store_true')
transcribe_parser.add_argument('-d', '--debug', help="Output debug messages about alignment", action='store_true')

annotator_parser = subparsers.add_parser('annotator')

thirdparty_parser = subparsers.add_parser('thirdparty')

thirdparty_parser.add_argument("command",
                               help="One of 'download', 'validate', or 'kaldi'")
thirdparty_parser.add_argument('local_directory',
                               help='Full path to the built executables to collect', nargs='?',
                               default='')


def main():
    mp.freeze_support()
    args, unknown = parser.parse_known_args()
    try:
        fix_path()
        if args.subcommand in ['align', 'train', 'train_ivector']:
            from montreal_forced_aligner.thirdparty.kaldi import validate_alignment_binaries
            if not validate_alignment_binaries():
                print("There was an issue validating Kaldi binaries, please ensure you've downloaded them via the "
                      "'mfa thirdparty download' command.  See 'mfa thirdparty validate' for more detailed information "
                      "on why this check failed.")
                sys.exit(1)
        elif args.subcommand in ['transcribe']:
            from montreal_forced_aligner.thirdparty.kaldi import validate_transcribe_binaries
            if not validate_transcribe_binaries():
                print("There was an issue validating Kaldi binaries, please ensure you've downloaded them via the "
                      "'mfa thirdparty download' command.  See 'mfa thirdparty validate' for more detailed information "
                      "on why this check failed.  If you are on MacOS, please note that the thirdparty binaries available "
                      "via the download command do not contain the transcription ones.  To get this functionality working "
                      "for the time being, please build kaldi locally and follow the instructions for running the "
                      "'mfa thirdparty kaldi' command.")
                sys.exit(1)
        elif args.subcommand in ['train_dictionary']:
            from montreal_forced_aligner.thirdparty.kaldi import validate_train_dictionary_binaries
            if not validate_train_dictionary_binaries():
                print("There was an issue validating Kaldi binaries, please ensure you've downloaded them via the "
                      "'mfa thirdparty download' command.  See 'mfa thirdparty validate' for more detailed information "
                      "on why this check failed.  If you are on MacOS, please note that the thirdparty binaries available "
                      "via the download command do not contain the train_dictionary ones.  To get this functionality working "
                      "for the time being, please build kaldi locally and follow the instructions for running the "
                      "'mfa thirdparty kaldi' command.")
                sys.exit(1)
        elif args.subcommand in ['g2p', 'train_g2p']:
            try:
                import pynini
            except ImportError:
                print("There was an issue importing Pynini, please ensure that it is installed. If you are on Windows, "
                      "please use the Windows Subsystem for Linux to use g2p functionality.")
                sys.exit(1)
        if args.subcommand == 'align':
            run_align_corpus(args, unknown, acoustic_languages)
        elif args.subcommand == 'adapt':
            run_adapt_model(args, unknown, acoustic_languages)
        elif args.subcommand == 'train':
            run_train_corpus(args, unknown)
        elif args.subcommand == 'g2p':
            run_g2p(args, unknown, g2p_languages)
        elif args.subcommand == 'train_g2p':
            run_train_g2p(args, unknown)
        elif args.subcommand == 'validate':
            run_validate_corpus(args, unknown)
        elif args.subcommand == 'download':
            run_download(args)
        elif args.subcommand == 'train_lm':
            run_train_lm(args, unknown)
        elif args.subcommand == 'train_dictionary':
            run_train_dictionary(args, unknown)
        elif args.subcommand == 'train_ivector':
            run_train_ivector_extractor(args, unknown)
        elif args.subcommand == 'classify_speakers':
            run_classify_speakers(args, unknown)
        elif args.subcommand == 'annotator':
            from montreal_forced_aligner.command_line.annotator import run_annotator
            run_annotator(args)
        elif args.subcommand == 'thirdparty':
            run_thirdparty(args)
        elif args.subcommand == 'transcribe':
            run_transcribe_corpus(args, unknown)
        elif args.subcommand == 'create_segments':
            run_create_segments(args, unknown)
        elif args.subcommand == 'version':
            print(__version__)
    except MFAError as e:
        if getattr(args, 'debug', False):
            raise
        print(e)
        sys.exit(1)
    finally:
        unfix_path()


if __name__ == '__main__':
    main()
