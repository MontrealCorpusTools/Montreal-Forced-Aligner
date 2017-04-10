import argparse
from aligner.g2p_trainer.train import Trainer


def train_g2p(path, path_to_dict, KO=False):
    path_to_model = Trainer(path, path_to_dict, KO).get_path_to_model()
    return path_to_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "train a g2p (grapheme to phoneme) model from an existing dictionary")

    parser.add_argument("--path", dest='path',
     required=True, help="desired location of generated model")

    parser.add_argument("--path_to_dict", dest='path_to_dict',
        required=True, help="location of existing dictionary")

    parser.add_argument("--KO", dest="KO", required=False, help="set to true if dictionary is in Korean. Decomposes Hangul into separate phonemes and increases accuracy")

    args  = parser.parse_args()

    path_to_model = train_g2p(args.path, args.path_to_dict, args.KO)
    