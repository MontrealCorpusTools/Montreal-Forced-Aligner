import sys
import os
import time
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
from montreal_forced_aligner.config import update_global_config, load_global_config, update_command_history

BEGIN = time.time()

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

def create_parser():

    GLOBAL_CONFIG = load_global_config()
    def add_global_options(subparser, textgrid_output=False):

        subparser.add_argument('-t', '--temp_directory', type=str, default=GLOBAL_CONFIG['temp_directory'],
                                  help=f"Temporary directory root to store MFA created files, default is {GLOBAL_CONFIG['temp_directory']}")
        subparser.add_argument('--disable_mp', help=f"Disable any multiprocessing during alignment (not recommended), default is {not GLOBAL_CONFIG['use_mp']}", action='store_true',
                                     default=not GLOBAL_CONFIG['use_mp'])
        subparser.add_argument('-j', '--num_jobs', type=int, default=GLOBAL_CONFIG['num_jobs'],
                                  help=f"Number of data splits (and cores to use if multiprocessing is enabled), defaults "
                                       f"is {GLOBAL_CONFIG['num_jobs']}")
        subparser.add_argument('-v', '--verbose', help=f"Output debug messages, default is {GLOBAL_CONFIG['verbose']}", action='store_true',
                                  default=GLOBAL_CONFIG['verbose'])
        subparser.add_argument('--clean', help=f"Remove files from previous runs, default is {GLOBAL_CONFIG['clean']}", action='store_true',
                                  default=GLOBAL_CONFIG['clean'])
        subparser.add_argument('--overwrite', help=f"Overwrite output files when they exist, default is {GLOBAL_CONFIG['overwrite']}", action='store_true',
                                  default=GLOBAL_CONFIG['overwrite'])
        subparser.add_argument('--debug', help=f"Run extra steps for debugging issues, default is {GLOBAL_CONFIG['debug']}", action='store_true',
                                  default=GLOBAL_CONFIG['debug'])
        if textgrid_output:
            subparser.add_argument('--disable_textgrid_cleanup', help=f"Disable extra clean up steps on TextGrid output, default is {not GLOBAL_CONFIG['cleanup_textgrids']}", action='store_true',
                                      default=not GLOBAL_CONFIG['cleanup_textgrids'])


    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="subcommand")
    subparsers.required = True

    version_parser = subparsers.add_parser('version')

    align_parser = subparsers.add_parser('align')
    align_parser.add_argument('corpus_directory', help="Full path to the directory to align")
    align_parser.add_argument('dictionary_path', help="Full path to the pronunciation dictionary to use")
    align_parser.add_argument('acoustic_model_path',
                              help=f"Full path to the archive containing pre-trained model or language ({', '.join(acoustic_languages)})")
    align_parser.add_argument('output_directory',
                              help="Full path to output directory, will be created if it doesn't exist")
    align_parser.add_argument('--config_path', type=str, default='',
                              help="Path to config file to use for alignment")
    align_parser.add_argument('-s', '--speaker_characters', type=str, default='0',
                              help="Number of characters of file names to use for determining speaker, "
                                   'default is to use directory names')
    align_parser.add_argument('-a', '--audio_directory', type=str, default='',
                                   help="Audio directory root to use for finding audio files")
    add_global_options(align_parser, textgrid_output=True)

    adapt_parser = subparsers.add_parser('adapt')
    adapt_parser.add_argument('corpus_directory', help="Full path to the directory to align")
    adapt_parser.add_argument('dictionary_path', help="Full path to the pronunciation dictionary to use")
    adapt_parser.add_argument('acoustic_model_path',
                              help=f"Full path to the archive containing pre-trained model or language ({', '.join(acoustic_languages)})")
    adapt_parser.add_argument('output_model_path',
                              help="Full path to save adapted_model")
    adapt_parser.add_argument('--config_path', type=str, default='',
                              help="Path to config file to use for alignment")
    adapt_parser.add_argument('-s', '--speaker_characters', type=str, default='0',
                              help="Number of characters of file names to use for determining speaker, "
                                   'default is to use directory names')
    adapt_parser.add_argument('-a', '--audio_directory', type=str, default='',
                                   help="Audio directory root to use for finding audio files")
    add_global_options(adapt_parser, textgrid_output=True)

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('corpus_directory', help="Full path to the source directory to align")
    train_parser.add_argument('dictionary_path', help="Full path to the pronunciation dictionary to use",
                              default='')
    train_parser.add_argument('output_directory',
                              help="Full path to output directory, will be created if it doesn't exist")
    train_parser.add_argument('--config_path', type=str, default='',
                              help="Path to config file to use for training and alignment")
    train_parser.add_argument('-o', '--output_model_path', type=str, default='',
                              help="Full path to save resulting acoustic and dictionary model")
    train_parser.add_argument('-s', '--speaker_characters', type=str, default='0',
                              help="Number of characters of filenames to use for determining speaker, "
                                   'default is to use directory names')
    train_parser.add_argument('-a', '--audio_directory', type=str, default='',
                                   help="Audio directory root to use for finding audio files")
    add_global_options(train_parser, textgrid_output=True)

    validate_parser = subparsers.add_parser('validate')
    validate_parser.add_argument('corpus_directory', help="Full path to the source directory to align")
    validate_parser.add_argument('dictionary_path', help="Full path to the pronunciation dictionary to use",
                                 default='')
    validate_parser.add_argument('acoustic_model_path',  nargs='?', default='',
                                 help=f"Full path to the archive containing pre-trained model or language ({', '.join(acoustic_languages)})")
    validate_parser.add_argument('-s', '--speaker_characters', type=str, default='0',
                                 help="Number of characters of file names to use for determining speaker, "
                                      'default is to use directory names')
    validate_parser.add_argument('--test_transcriptions', help="Test accuracy of transcriptions", action='store_true')
    validate_parser.add_argument('--ignore_acoustics',
                                 help="Skip acoustic feature generation and associated validation",
                                 action='store_true')
    add_global_options(validate_parser)

    g2p_model_help_message = f'''Full path to the archive containing pre-trained model or language ({', '.join(g2p_languages)})
    If not specified, then orthographic transcription is split into pronunciations.'''
    g2p_parser = subparsers.add_parser('g2p')
    g2p_parser.add_argument("g2p_model_path", help=g2p_model_help_message, nargs='?')

    g2p_parser.add_argument("input_path",
                            help="Corpus to base word list on or a text file of words to generate pronunciations")
    g2p_parser.add_argument("output_path", help="Path to save output dictionary")
    g2p_parser.add_argument('--include_bracketed', help="Included words enclosed by brackets, i.e. [...], (...), <...>",
                            action='store_true')
    g2p_parser.add_argument('--config_path', type=str, default='',
                              help="Path to config file to use for G2P")
    add_global_options(g2p_parser)

    train_g2p_parser = subparsers.add_parser('train_g2p')
    train_g2p_parser.add_argument("dictionary_path", help="Location of existing dictionary")

    train_g2p_parser.add_argument("output_model_path", help="Desired location of generated model")
    train_g2p_parser.add_argument('--config_path', type=str, default='',
                              help="Path to config file to use for G2P")
    train_g2p_parser.add_argument("--validate", action='store_true',
                                  help="Perform an analysis of accuracy training on "
                                       "most of the data and validating on an unseen subset")
    add_global_options(train_g2p_parser)

    download_parser = subparsers.add_parser('download')
    download_parser.add_argument("model_type",
                                 help="Type of model to download, one of 'acoustic', 'g2p', or 'dictionary'")
    download_parser.add_argument("language", help="Name of language code to download, if not specified, "
                                                  "will list all available languages", nargs='?')

    train_lm_parser = subparsers.add_parser('train_lm')
    train_lm_parser.add_argument('source_path', help="Full path to the source directory to train from, alternatively "
                                                     'an ARPA format language model to convert for MFA use')
    train_lm_parser.add_argument('output_model_path', type=str,
                                 help="Full path to save resulting language model")
    train_lm_parser.add_argument('-m', '--model_path', type=str,
                                 help="Full path to existing language model to merge probabilities")
    train_lm_parser.add_argument('-w', '--model_weight', type=float, default=1.0,
                                 help="Weight factor for supplemental language model, defaults to 1.0")
    train_lm_parser.add_argument('--dictionary_path', help="Full path to the pronunciation dictionary to use",
                                 default='')
    train_lm_parser.add_argument('--config_path', type=str, default='',
                                 help="Path to config file to use for training and alignment")
    add_global_options(train_lm_parser)

    train_dictionary_parser = subparsers.add_parser('train_dictionary')
    train_dictionary_parser.add_argument('corpus_directory', help="Full path to the directory to align")
    train_dictionary_parser.add_argument('dictionary_path', help="Full path to the pronunciation dictionary to use")
    train_dictionary_parser.add_argument('acoustic_model_path',
                                         help=f"Full path to the archive containing pre-trained model or language ({', '.join(acoustic_languages)})")
    train_dictionary_parser.add_argument('output_directory',
                                         help="Full path to output directory, will be created if it doesn't exist")
    train_dictionary_parser.add_argument('--config_path', type=str, default='',
                                         help="Path to config file to use for alignment")
    train_dictionary_parser.add_argument('-s', '--speaker_characters', type=str, default='0',
                                         help="Number of characters of file names to use for determining speaker, "
                                              'default is to use directory names')
    add_global_options(train_dictionary_parser)


    train_ivector_parser = subparsers.add_parser('train_ivector')
    train_ivector_parser.add_argument('corpus_directory', help="Full path to the source directory to "
                                                               'train the ivector extractor')
    train_ivector_parser.add_argument('dictionary_path', help="Full path to the pronunciation dictionary to use")
    train_ivector_parser.add_argument('acoustic_model_path', type=str, default='',
                                      help="Full path to acoustic model for alignment")
    train_ivector_parser.add_argument('output_model_path', type=str, default='',
                                      help="Full path to save resulting ivector extractor")
    train_ivector_parser.add_argument('-s', '--speaker_characters', type=str, default='0',
                                      help="Number of characters of filenames to use for determining speaker, "
                                           'default is to use directory names')
    train_ivector_parser.add_argument('--config_path', type=str, default='',
                                      help="Path to config file to use for training")
    add_global_options(train_ivector_parser)

    classify_speakers_parser = subparsers.add_parser('classify_speakers')
    classify_speakers_parser.add_argument('corpus_directory', help="Full path to the source directory to "
                                                                   'run speaker classification')
    classify_speakers_parser.add_argument('ivector_extractor_path', type=str, default='',
                                          help="Full path to ivector extractor model")
    classify_speakers_parser.add_argument('output_directory',
                                          help="Full path to output directory, will be created if it doesn't exist")

    classify_speakers_parser.add_argument('-s', '--num_speakers', type=int, default=0,
                                          help="Number of speakers if known")
    classify_speakers_parser.add_argument('--cluster', help="Using clustering instead of classification", action='store_true')
    classify_speakers_parser.add_argument('--config_path', type=str, default='',
                                          help="Path to config file to use for ivector extraction")
    add_global_options(classify_speakers_parser)

    create_segments_parser = subparsers.add_parser('create_segments')
    create_segments_parser.add_argument('corpus_directory', help="Full path to the source directory to "
                                                                   'run VAD segmentation')
    create_segments_parser.add_argument('output_directory',
                                          help="Full path to output directory, will be created if it doesn't exist")
    create_segments_parser.add_argument('--config_path', type=str, default='',
                                          help="Path to config file to use for segmentation")
    add_global_options(create_segments_parser)

    transcribe_parser = subparsers.add_parser('transcribe')
    transcribe_parser.add_argument('corpus_directory', help="Full path to the directory to transcribe")
    transcribe_parser.add_argument('dictionary_path', help="Full path to the pronunciation dictionary to use")
    transcribe_parser.add_argument('acoustic_model_path',
                                   help=f"Full path to the archive containing pre-trained model or language ({', '.join(acoustic_languages)})")
    transcribe_parser.add_argument('language_model_path',
                                   help=f"Full path to the archive containing pre-trained model or language ({', '.join(lm_languages)})")
    transcribe_parser.add_argument('output_directory',
                                   help="Full path to output directory, will be created if it doesn't exist")
    transcribe_parser.add_argument('--config_path', type=str, default='',
                                   help="Path to config file to use for transcription")
    transcribe_parser.add_argument('-s', '--speaker_characters', type=str, default='0',
                                   help="Number of characters of file names to use for determining speaker, "
                                        'default is to use directory names')
    transcribe_parser.add_argument('-a', '--audio_directory', type=str, default='',
                                   help="Audio directory root to use for finding audio files")
    transcribe_parser.add_argument('-e', '--evaluate', help="Evaluate the transcription "
                                                            "against golden texts", action='store_true')
    add_global_options(transcribe_parser)

    config_parser = subparsers.add_parser('configure', help="The configure command is used to set global defaults for MFA so "
                                                            "you don't have to set them every time you call an MFA command.")
    config_parser.add_argument('-t', '--temp_directory', type=str, default='',
                                   help=f"Set the default temporary directory, default is {GLOBAL_CONFIG['temp_directory']}")
    config_parser.add_argument('-j', '--num_jobs', type=int,
                                          help=f"Set the number of processes to use by default, defaults to {GLOBAL_CONFIG['num_jobs']}")
    config_parser.add_argument('--always_clean', help="Always remove files from previous runs by default", action='store_true')
    config_parser.add_argument('--never_clean', help="Don't remove files from previous runs by default", action='store_true')
    config_parser.add_argument('--always_verbose', help="Default to verbose output", action='store_true')
    config_parser.add_argument('--never_verbose', help="Default to non-verbose output", action='store_true')
    config_parser.add_argument('--always_debug', help="Default to running debugging steps", action='store_true')
    config_parser.add_argument('--never_debug', help="Default to not running debugging steps", action='store_true')
    config_parser.add_argument('--always_overwrite', help="Always overwrite output files", action='store_true')
    config_parser.add_argument('--never_overwrite', help="Never overwrite output files (if file already exists, "
                                                         "the output will be saved in the temp directory)", action='store_true')
    config_parser.add_argument('--disable_mp', help="Disable all multiprocessing (not recommended as it will usually "
                                                    "increase processing times)", action='store_true')
    config_parser.add_argument('--enable_mp', help="Enable multiprocessing (recommended and enabled by default)",
                               action='store_true')
    config_parser.add_argument('--disable_textgrid_cleanup', help="Disable postprocessing of TextGrids that cleans up "
                                                                  "silences and recombines compound words and clitics", action='store_true')
    config_parser.add_argument('--enable_textgrid_cleanup', help="Enable postprocessing of TextGrids that cleans up "
                                                                  "silences and recombines compound words and clitics",
                               action='store_true')

    annotator_parser = subparsers.add_parser('annotator')
    anchor_parser = subparsers.add_parser('anchor')

    thirdparty_parser = subparsers.add_parser('thirdparty')

    thirdparty_parser.add_argument("command",
                                   help="One of 'download', 'validate', or 'kaldi'")
    thirdparty_parser.add_argument('local_directory',
                                   help="Full path to the built executables to collect", nargs="?",
                                   default='')
    return parser

parser = create_parser()


def main():

    parser = create_parser()
    mp.freeze_support()
    args, unknown = parser.parse_known_args()
    for short in ['-c', '-d']:
        if short in unknown:
            print(f'Due to the number of options that `{short}` could refer to, it is not accepted. '
                  'Please specify the full argument')
            sys.exit(1)
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
        elif args.subcommand in ['annotator', 'anchor']:
            from montreal_forced_aligner.command_line.anchor import run_anchor
            run_anchor(args)
        elif args.subcommand == 'thirdparty':
            run_thirdparty(args)
        elif args.subcommand == 'transcribe':
            run_transcribe_corpus(args, unknown)
        elif args.subcommand == 'create_segments':
            run_create_segments(args, unknown)
        elif args.subcommand == 'configure':
            update_global_config(args)
            global GLOBAL_CONFIG
            GLOBAL_CONFIG = load_global_config()
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
