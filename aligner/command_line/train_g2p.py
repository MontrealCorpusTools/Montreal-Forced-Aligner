import argparse
import os
from ..g2p.trainer import PhonetisaurusTrainer

from ..dictionary import Dictionary
from .. exceptions import ArgumentError


def train_g2p(args):
    dictionary = Dictionary(args.dictionary_path, '')
    t = PhonetisaurusTrainer(dictionary, args.path, temp_directory=args.temp_directory, korean=args.korean)

    t.train()


def validate(args):
    if not os.path.exists(args.dictionary_path):
        raise (ArgumentError('Could not find the dictionary file {}'.format(args.dictionary_path)))
    if not os.path.isfile(args.dictionary_path):
        raise (ArgumentError('The specified dictionary path ({}) is not a text file.'.format(args.dictionary_path)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a g2p (grapheme to phoneme) model from an existing dictionary")

    parser.add_argument("--dictionary_path",
                        required=True, help="Location of existing dictionary")

    parser.add_argument("--path",
                        required=True, help="Desired location of generated model")
    parser.add_argument('-t', '--temp_directory', type=str, default='',
                        help='Temporary directory root to use for G2P training, default is ~/Documents/MFA')

    parser.add_argument("--korean", action='store_true',
                        help="Set to true if dictionary is in Korean. "
                             "Decomposes Hangul into separate letters (jamo) and increases accuracy")

    args = parser.parse_args()
    validate(args)
    path_to_model = train_g2p(args)
