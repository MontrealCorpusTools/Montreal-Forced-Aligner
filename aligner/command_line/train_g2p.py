import argparse
import os
from aligner.g2p.train import Trainer


def train_g2p(args):
    t = Trainer(args.dictionary_path, args.path, korean = args.KO)

    return path_to_model


def validate(args):
    if not os.path.exists(args.dictionary_path):
        raise(FileNotFoundError('Could not find the dictionary file {}'.format(args.dictionary_path)))
    if not os.path.isfile(args.dictionary_path):
        raise(Exception('The specified dictionary path ({}) is not a text file.'.format(args.dictionary_path)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a g2p (grapheme to phoneme) model from an existing dictionary")

    parser.add_argument("--dictionary_path",
                        required=True, help="Location of existing dictionary")

    parser.add_argument("--path",
                        required=True, help="Desired location of generated model")

    parser.add_argument("--KO", action='store_true',
                        help="Set to true if dictionary is in Korean. "
                             "Decomposes Hangul into separate letters (jamo) and increases accuracy")

    args = parser.parse_args()
    validate(args)
    path_to_model = train_g2p(args)
