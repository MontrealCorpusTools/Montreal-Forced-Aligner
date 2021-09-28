import os
import shutil
from montreal_forced_aligner.g2p.trainer import PyniniTrainer as Trainer
from montreal_forced_aligner.dictionary import Dictionary
from montreal_forced_aligner.exceptions import ArgumentError
from montreal_forced_aligner.config import TEMP_DIR
from montreal_forced_aligner.config.train_g2p_config import train_g2p_yaml_to_config, load_basic_train_g2p_config
from montreal_forced_aligner.utils import get_available_dict_languages, get_dictionary_path


def train_g2p(args, unknown_args=None):
    if not args.temp_directory:
        temp_dir = TEMP_DIR
    else:
        temp_dir = os.path.expanduser(args.temp_directory)
    if args.clean:
        shutil.rmtree(os.path.join(temp_dir, 'G2P'), ignore_errors=True)
        shutil.rmtree(os.path.join(temp_dir, 'models', 'G2P'), ignore_errors=True)
    if args.config_path:
        train_config = train_g2p_yaml_to_config(args.config_path)
    else:
        train_config = load_basic_train_g2p_config()
    train_config.use_mp = not args.disable_mp
    if unknown_args:
        train_config.update_from_args(unknown_args)
    dictionary = Dictionary(args.dictionary_path, '')
    t = Trainer(dictionary, args.output_model_path, temp_directory=temp_dir, train_config=train_config, num_jobs=args.num_jobs,
                verbose=args.verbose)
    if args.validate:
        t.validate()
    else:
        t.train()


def validate(args, download_dictionaries=None):
    if args.dictionary_path.lower() in download_dictionaries:
        args.dictionary_path = get_dictionary_path(args.dictionary_path.lower())
    if not os.path.exists(args.dictionary_path):
        raise (ArgumentError('Could not find the dictionary file {}'.format(args.dictionary_path)))
    if not os.path.isfile(args.dictionary_path):
        raise (ArgumentError('The specified dictionary path ({}) is not a text file.'.format(args.dictionary_path)))


def run_train_g2p(args, unknown, download_dictionaries=None):
    if download_dictionaries is None:
        download_dictionaries = get_available_dict_languages()
    validate(args, download_dictionaries)
    train_g2p(args, unknown)

