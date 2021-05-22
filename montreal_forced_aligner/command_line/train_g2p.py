import os
import shutil
from montreal_forced_aligner.g2p.trainer import PyniniTrainer as Trainer
from montreal_forced_aligner.dictionary import Dictionary
from montreal_forced_aligner.exceptions import ArgumentError
from montreal_forced_aligner.config import TEMP_DIR
from montreal_forced_aligner.utils import get_available_dict_languages, get_dictionary_path


def train_g2p(args):
    if not args.temp_directory:
        temp_dir = TEMP_DIR
    else:
        temp_dir = os.path.expanduser(args.temp_directory)
    if args.clean:
        shutil.rmtree(os.path.join(temp_dir, 'G2P'), ignore_errors=True)
        shutil.rmtree(os.path.join(temp_dir, 'models', 'G2P'), ignore_errors=True)
    dictionary = Dictionary(args.dictionary_path, '')
    t = Trainer(dictionary, args.output_model_path, temp_directory=temp_dir, order=args.order, num_jobs=args.num_jobs,
                use_mp=not args.disable_mp, verbose=args.verbose)
    if args.validate:
        t.validate()
    t.train()


def validate(args, download_dictionaries=None):
    if args.dictionary_path.lower() in download_dictionaries:
        args.dictionary_path = get_dictionary_path(args.dictionary_path.lower())
    if not os.path.exists(args.dictionary_path):
        raise (ArgumentError('Could not find the dictionary file {}'.format(args.dictionary_path)))
    if not os.path.isfile(args.dictionary_path):
        raise (ArgumentError('The specified dictionary path ({}) is not a text file.'.format(args.dictionary_path)))


def run_train_g2p(args, download_dictionaries=None):
    if download_dictionaries is None:
        download_dictionaries = get_available_dict_languages()
    validate(args, download_dictionaries)
    train_g2p(args)


if __name__ == '__main__':  # pragma: no cover
    from montreal_forced_aligner.command_line.mfa import train_g2p_parser, fix_path, unfix_path, dict_languages

    train_args = train_g2p_parser.parse_args()
    fix_path()
    run_train_g2p(train_args, dict_languages)
    unfix_path()
