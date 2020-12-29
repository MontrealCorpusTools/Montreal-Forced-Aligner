import os
import multiprocessing as mp

from montreal_forced_aligner.corpus.align_corpus import AlignableCorpus
from montreal_forced_aligner.dictionary import Dictionary

from montreal_forced_aligner.config import TEMP_DIR, train_lm_yaml_to_config, load_basic_train_lm

from montreal_forced_aligner.exceptions import ArgumentError

from montreal_forced_aligner.lm.trainer import LmTrainer
from montreal_forced_aligner.utils import get_available_dict_languages, get_dictionary_path


def train_lm(args):
    if not args.temp_directory:
        temp_dir = TEMP_DIR
    else:
        temp_dir = os.path.expanduser(args.temp_directory)
    corpus_name = os.path.basename(args.corpus_directory)
    if corpus_name == '':
        args.corpus_directory = os.path.dirname(args.corpus_directory)
        corpus_name = os.path.basename(args.corpus_directory)

    data_directory = os.path.join(temp_dir, corpus_name)
    corpus = AlignableCorpus(args.corpus_directory, data_directory)
    if args.config_path:
        train_config = train_lm_yaml_to_config(args.config_path)
    else:
        train_config = load_basic_train_lm()
    if args.dictionary_path is not None:
        dictionary = Dictionary(args.dictionary_path, data_directory)
    else:
        dictionary = None
    trainer = LmTrainer(corpus, train_config, args.output_model_path, dictionary=dictionary,
                        temp_directory=data_directory, num_jobs=args.num_jobs,
                        supplemental_model_path=args.model_path, supplemental_model_weight=args.model_weight)
    trainer.train()


def validate_args(args, download_dictionaries=None):
    if args.dictionary_path is not None and args.dictionary_path.lower() in download_dictionaries:
        args.dictionary_path = get_dictionary_path(args.dictionary_path.lower())
    if not os.path.exists(args.corpus_directory):
        raise (ArgumentError('Could not find the corpus directory {}.'.format(args.corpus_directory)))
    if not os.path.isdir(args.corpus_directory):
        raise (ArgumentError('The specified corpus directory ({}) is not a directory.'.format(args.corpus_directory)))
    if args.config_path and not os.path.exists(args.config_path):
        raise (ArgumentError('Could not find the config file {}.'.format(args.config_path)))
    if args.model_path and not os.path.exists(args.model_path):
        raise (ArgumentError('Could not find the model file {}.'.format(args.model_path)))


def run_train_lm(args, download_dictionaries=None):
    if not args.dictionary_path:
        args.dictionary_path = None
    if download_dictionaries is None:
        download_dictionaries = get_available_dict_languages()
    args.corpus_directory = args.corpus_directory.rstrip('/').rstrip('\\')

    validate_args(args, download_dictionaries)
    train_lm(args)


if __name__ == '__main__':  # pragma: no cover
    mp.freeze_support()
    from montreal_forced_aligner.command_line.mfa import train_lm_parser, fix_path, unfix_path, dict_languages
    args = train_lm_parser.parse_args()

    fix_path()
    run_train_lm(args, dict_languages)
    unfix_path()