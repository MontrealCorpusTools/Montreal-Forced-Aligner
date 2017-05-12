import argparse
import os

from aligner.g2p.generator import PhonetisaurusDictionaryGenerator
from aligner.corpus import Corpus
from aligner.models import G2PModel

from aligner.exceptions import ArgumentError
from aligner.config import TEMP_DIR
from aligner.command_line.align import fix_path, unfix_path


def generate_dict(args):
    if not args.temp_directory:
        temp_dir = TEMP_DIR
    else:
        temp_dir = os.path.expanduser(args.temp_directory)
    input_dir = os.path.expanduser(args.corpus_directory)

    corpus = Corpus(input_dir, os.path.join(temp_dir, 'corpus'))
    word_set = corpus.word_set
    model = G2PModel(args.g2p_model_path)

    gen = PhonetisaurusDictionaryGenerator(model, word_set, args.output_path, temp_directory=temp_dir)
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

    parser.add_argument("g2p_model_path", help="Path to the trained G2P model")

    parser.add_argument("corpus_directory", help="Corpus to base word list on")

    parser.add_argument("output_path", help="Path to save output dictionary")

    parser.add_argument('-t', '--temp_directory', type=str, default='',
                        help='Temporary directory root to use for dictionary generation, default is ~/Documents/MFA')

    args = parser.parse_args()
    fix_path()

    validate(args)
    generate_dict(args)
    unfix_path()

