import argparse
import os
import sys
import traceback

from textgrid import TextGrid, IntervalTier

from aligner.corpus import load_text, parse_transcription

from aligner.g2p.generator import PhonetisaurusDictionaryGenerator
from aligner.corpus import Corpus
from aligner.models import G2PModel
from aligner.dictionary import check_bracketed

from aligner.exceptions import ArgumentError
from aligner.config import TEMP_DIR

from aligner.command_line.align import fix_path, unfix_path


def generate_g2p_dict(args):
    print("Generating pronunciations from G2P model")
    if not args.temp_directory:
        temp_dir = TEMP_DIR
        temp_dir = os.path.join(temp_dir, 'G2P')
    else:
        temp_dir = os.path.expanduser(args.temp_directory)
    if os.path.isdir(args.input_path):
        input_dir = os.path.expanduser(args.input_path)
        corpus_name = os.path.basename(args.input_path)
        if corpus_name == '':
            args.input_path = os.path.dirname(args.input_path)
            corpus_name = os.path.basename(args.input_path)
        data_directory = os.path.join(temp_dir, corpus_name)

        corpus = Corpus(input_dir, data_directory)

        word_set = get_word_set(corpus, args.include_bracketed)

    else:
        word_set = set()
        with open(args.input_path, 'r', encoding='utf8') as f:
            for line in f:
                word_set.update(line.strip().split())
    model = G2PModel(args.g2p_model_path)
    gen = PhonetisaurusDictionaryGenerator(model, word_set, args.output_path, temp_directory=temp_dir)
    gen.generate()


def generate_orthography_dict(args):
    print("Generating pronunciations from orthography")
    if not args.temp_directory:
        temp_dir = TEMP_DIR
        temp_dir = os.path.join(temp_dir, 'G2P')
    else:
        temp_dir = os.path.expanduser(args.temp_directory)

    if os.path.isdir(args.input_path):
        input_dir = os.path.expanduser(args.input_path)
        corpus_name = os.path.basename(args.input_path)
        if corpus_name == '':
            args.input_path = os.path.dirname(args.input_path)
            corpus_name = os.path.basename(args.input_path)
        data_directory = os.path.join(temp_dir, corpus_name)

        corpus = Corpus(input_dir, data_directory)

        word_set = get_word_set(corpus, args.include_bracketed)

    else:
        word_set = set()
        with open(args.input_path, 'r', encoding='utf8') as f:
            for line in f:
                word_set.update(line.strip().split())
        if not args.include_bracketed:
            word_set = [x for x in word_set if not check_bracketed(x)]

    with open(args.output_path, "w", encoding='utf8') as f:
        for word in sorted(word_set):
            pronunciation = list(word)
            f.write('{} {}\n'.format(word, ' '.join(pronunciation)))


def get_word_set(corpus, include_bracketed=False):
    word_set = corpus.word_set
    decode_error_files = []
    textgrid_read_errors = {}
    for file_path in corpus.transcriptions_without_wavs:
        if file_path.endswith('.lab'):
            try:
                text = load_text(file_path)
            except UnicodeDecodeError:
                decode_error_files.append(file_path)
                continue
            words = parse_transcription(text)
            word_set.update(words)
        else:
            tg = TextGrid()
            try:
                tg.read(file_path)
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                textgrid_read_errors[file_path] = '\n'.join(
                    traceback.format_exception(exc_type, exc_value, exc_traceback))
                continue
            for i, ti in enumerate(tg.tiers):
                if ti.name.lower() == 'notes':
                    continue
                if not isinstance(ti, IntervalTier):
                    continue
                for interval in ti:
                    text = interval.mark.lower().strip()
                    words = parse_transcription(text)
                    if not words:
                        continue
                    word_set.update(words)

    if decode_error_files:
        print('WARNING: The following files were not able to be decoded using utf8:\n\n'
              '{}'.format('\n'.join(decode_error_files)))
    if textgrid_read_errors:
        print('WARNING: The following TextGrid files were not able to be read:\n\n'
              '{}'.format('\n'.join(textgrid_read_errors.keys())))
    print('Generating transcriptions for the {} word types found in the corpus...'.format(len(word_set)))
    if not include_bracketed:
        word_set = [x for x in word_set if not check_bracketed(x)]
    return word_set


def validate(args):
    if not args.g2p_model_path:
        args.g2p_model_path = None
    if args.g2p_model_path and not os.path.exists(args.g2p_model_path):
        raise (ArgumentError('Could not find the G2P model file {}.'.format(args.g2p_model_path)))
    if args.g2p_model_path and (not os.path.isfile(args.g2p_model_path) or not args.g2p_model_path.endswith('.zip')):
        raise (ArgumentError('The specified G2P model path ({}) is not a zip file.'.format(args.g2p_model_path)))

    if not os.path.exists(args.input_path):
        raise (ArgumentError('Could not find the input path {}.'.format(args.input_path)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a dictionary from a G2P model (or from orthography)")

    parser.add_argument("g2p_model_path", help="Path to the trained G2P model", nargs='?')

    parser.add_argument("input_path",
                        help="Corpus to base word list on or a text file of words to generate pronunciations")

    parser.add_argument("output_path", help="Path to save output dictionary")

    parser.add_argument('-t', '--temp_directory', type=str, default='',
                        help='Temporary directory root to use for dictionary generation, default is ~/Documents/MFA')
    parser.add_argument('--include_bracketed', help="Included words enclosed by brackets, i.e. [...], (...), <...>",
                        action='store_true')
    args = parser.parse_args()

    fix_path()
    validate(args)
    if args.g2p_model_path is not None:
        generate_g2p_dict(args)
    else:
        generate_orthography_dict(args)
    unfix_path()
