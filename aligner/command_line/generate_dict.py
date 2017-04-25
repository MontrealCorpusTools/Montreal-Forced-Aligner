import argparse
import os

from ..g2p.generator import PhonetisaurusDictionaryGenerator
from ..corpus import Corpus
from ..models import G2PModel

from ..exceptions import ArgumentError


def generate_dict(args):
    input_dir = os.path.expanduser(args.corpus_directory)
    corpus = Corpus(input_dir, "")

    model = G2PModel(args.g2p_model_path)

    gen = PhonetisaurusDictionaryGenerator(model, corpus, args.output_path, temp_directory=args.temp_directory,
                                         korean=args.korean)
    gen.generate()


def validate(args):
    if not os.path.exists(args.g2p_model_path):
        raise (ArgumentError('Could not find the G2P model file {}.'.format(args.g2p_model_path)))
    if not os.path.isfile(args.g2p_model_path) or not args.g2p_model_path.endswith('.zip'):
        raise (ArgumentError('The specified G2P model path ({}) is not a zip file.'.format(args.g2p_model_path)))

    if not os.path.exists(args.corpus_directory):
        raise (ArgumentError('Could not find the corpus directory {}.'.format(args.corpus_directory)))
    if not os.path.isdir(args.corpus_directory):
        raise (ArgumentError('The specified corpus directory ({}) is not a directory.'.format(args.g2p_model_path)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a dictionary from a G2P model")

    parser.add_argument("--g2p_model_path",
                        required=True, help="Path to the trained G2P model")

    parser.add_argument("--corpus_directory",
                        required=True, help="Corpus to base word list on")

    parser.add_argument("--output_path",
                        help="Path to save output dictionary")

    parser.add_argument('-t', '--temp_directory', type=str, default='',
                        help='Temporary directory root to use for dictionary generation, default is ~/Documents/MFA')

    parser.add_argument("--korean", action='store_true',
                        help="Set to true if corpus is in Korean. "
                             "Decomposes Hangul into separate letters (jamo) and increases accuracy")

    args = parser.parse_args()
    validate(args)
    generate_dict(args)
