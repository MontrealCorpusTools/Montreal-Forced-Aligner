import argparse
import os
from aligner.g2p.trainer import PhonetisaurusTrainer

from aligner.dictionary import Dictionary
from aligner.exceptions import ArgumentError
from aligner.config import TEMP_DIR

from aligner.command_line.align import fix_path, unfix_path

def train_g2p(args):
    if not args.temp_directory:
        temp_dir = TEMP_DIR
    else:
        temp_dir = os.path.expanduser(args.temp_directory)
    dictionary = Dictionary(args.dictionary_path, '')
    t = PhonetisaurusTrainer(dictionary, args.output_model_path, temp_directory=temp_dir, window_size=args.window_size)
    if args.validate:
        t.validate()
    t.train()


def validate(args):
    if not os.path.exists(args.dictionary_path):
        raise (ArgumentError('Could not find the dictionary file {}'.format(args.dictionary_path)))
    if not os.path.isfile(args.dictionary_path):
        raise (ArgumentError('The specified dictionary path ({}) is not a text file.'.format(args.dictionary_path)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a g2p (grapheme to phoneme) model from an existing dictionary")

    parser.add_argument("dictionary_path", help="Location of existing dictionary")

    parser.add_argument("output_model_path", help="Desired location of generated model")
    parser.add_argument('-t', '--temp_directory', type=str, default='',
                        help='Temporary directory root to use for G2P training, default is ~/Documents/MFA')

    parser.add_argument("--window_size", type=int, default=2,
                        help="Window size of phonemes, default is 2, increase if one character often corresponds to"
                             "more than 2 phonemes")
    parser.add_argument('-v', "--validate", action='store_true',
                        help="Perform an analysis of accuracy training on "
                             "most of the data and validating on an unseen subset")

    args = parser.parse_args()
    fix_path()
    validate(args)
    train_g2p(args)
    unfix_path()
