import os
import shutil

from montreal_forced_aligner.corpus.align_corpus import AlignableCorpus
from montreal_forced_aligner.config.g2p_config import load_basic_g2p_config, g2p_yaml_to_config

from montreal_forced_aligner.g2p.generator import PyniniDictionaryGenerator as Generator
from montreal_forced_aligner.models import G2PModel
from montreal_forced_aligner.dictionary import check_bracketed
from montreal_forced_aligner.utils import get_pretrained_g2p_path, get_available_g2p_languages

from montreal_forced_aligner.exceptions import ArgumentError
from montreal_forced_aligner.config import TEMP_DIR


def generate_dictionary(args, unknown_args=None):
    print("Generating pronunciations from G2P model")
    if not args.temp_directory:
        temp_dir = TEMP_DIR
        temp_dir = os.path.join(temp_dir, 'G2P')
    else:
        temp_dir = os.path.expanduser(args.temp_directory)
    if args.clean:
        shutil.rmtree(os.path.join(temp_dir, 'G2P'), ignore_errors=True)
        shutil.rmtree(os.path.join(temp_dir, 'models', 'G2P'), ignore_errors=True)
    if args.config_path:
        g2p_config = g2p_yaml_to_config(args.config_path)
    else:
        g2p_config = load_basic_g2p_config()
    g2p_config.use_mp = not args.disable_mp
    if unknown_args:
        g2p_config.update_from_args(unknown_args)
    if os.path.isdir(args.input_path):
        input_dir = os.path.expanduser(args.input_path)
        corpus_name = os.path.basename(args.input_path)
        if corpus_name == '':
            args.input_path = os.path.dirname(args.input_path)
            corpus_name = os.path.basename(args.input_path)
        data_directory = os.path.join(temp_dir, corpus_name)

        corpus = AlignableCorpus(input_dir, data_directory, num_jobs=args.num_jobs, use_mp=g2p_config.use_mp,
                                 punctuation=g2p_config.punctuation, clitic_markers=g2p_config.clitic_markers,
                                 parse_text_only_files=True)

        word_set = get_word_set(corpus, args.include_bracketed)
    else:
        word_set = []
        with open(args.input_path, 'r', encoding='utf8') as f:
            for line in f:
                word_set.extend(line.strip().split())
        if not args.include_bracketed:
            word_set = [x for x in word_set if not check_bracketed(x)]

    if args.g2p_model_path is not None:
        model = G2PModel(args.g2p_model_path, root_directory=os.path.join(temp_dir, 'models', 'G2P'))
        model.validate(word_set)
        num_jobs = args.num_jobs
        if not g2p_config.use_mp:
            num_jobs = 1
        gen = Generator(model, word_set, temp_directory=temp_dir, num_jobs=num_jobs,
                        num_pronunciations=g2p_config.num_pronunciations)
        gen.output(args.output_path)
        model.clean_up()
    else:
        with open(args.output_path, "w", encoding='utf8') as f:
            for word in word_set:
                pronunciation = list(word)
                f.write('{} {}\n'.format(word, ' '.join(pronunciation)))


def get_word_set(corpus, include_bracketed=False):
    word_set = corpus.word_set
    print('Generating transcriptions for the {} word types found in the corpus...'.format(len(word_set)))
    if not include_bracketed:
        word_set = [x for x in word_set if not check_bracketed(x)]
    return word_set


def validate(args, pretrained_languages):
    if not args.g2p_model_path:
        args.g2p_model_path = None
    elif args.g2p_model_path in pretrained_languages:
        args.g2p_model_path = get_pretrained_g2p_path(args.g2p_model_path)
    if args.g2p_model_path and not os.path.exists(args.g2p_model_path):
        raise (ArgumentError('Could not find the G2P model file {}.'.format(args.g2p_model_path)))
    if args.g2p_model_path and (not os.path.isfile(args.g2p_model_path) or not args.g2p_model_path.endswith('.zip')):
        raise (ArgumentError('The specified G2P model path ({}) is not a zip file.'.format(args.g2p_model_path)))

    if not os.path.exists(args.input_path):
        raise (ArgumentError('Could not find the input path {}.'.format(args.input_path)))


def run_g2p(args, unknown=None, pretrained=None):
    if pretrained is None:
        pretrained = get_available_g2p_languages()
    validate(args, pretrained)
    generate_dictionary(args, unknown)
