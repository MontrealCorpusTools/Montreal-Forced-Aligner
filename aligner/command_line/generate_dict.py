import argparse
from aligner.dictmaker.makedict import DictMaker


def generate_dict(language, input_dir, outfile, KO = None):
    input_dir = os.path.expanduser(input_dir)
    # corpus = Corpus(input_dir, input_dir, 0)
    # word_list = corpus.word_set()



    D = DictMaker(language, input_dir, outfile, KO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "create a dictionary from pre-built models for any of the following languages:")

    parser.add_argument("--language", dest='language',
     required=True, help="this is the language for which you'd like to create a dictionary. Current options: BG (Bulgarian), CH (Mandarin), CZ (Czech), PL (Polish), RU (Russian), SA (Swahili), UA (Ukrainian), VN (Vietnamese)")

    parser.add_argument("--input_dir", dest='input_dir',
        required=True, help="the location of your .lab files")
    
    parser.add_argument("--outfile", dest='outfile',
        required=False, help="the name of the output file for your dictionary (LANG_dict.txt by default)")

    parser.add_argument("--KO", dest="KO", required=False, help="add if your dictionary is in Korean. This will decompose the Hangul into separate phonemes and increase accuracy")

    args  = parser.parse_args()

    generate_dict(args.language, args.input_dir, args.outfile, args.KO)
    