import os
import pytest
from montreal_forced_aligner.corpus import AlignableCorpus
from montreal_forced_aligner.dictionary import Dictionary

def test_save_text_lab(basic_dict_path, basic_corpus_dir, generated_dir, default_feature_config):
    dictionary = Dictionary(basic_dict_path, os.path.join(generated_dir, 'basic'))
    dictionary.write()
    output_directory = os.path.join(generated_dir, 'basic')
    c = AlignableCorpus(basic_corpus_dir, output_directory, use_mp=True)
    c.initialize_corpus(dictionary)
    c.files['acoustic_corpus'].save()


def test_flac_tg(basic_dict_path, flac_tg_corpus_dir, temp_dir, default_feature_config):
    temp = os.path.join(temp_dir, 'flac_tg_corpus')
    dictionary = Dictionary(basic_dict_path, os.path.join(temp, 'basic'))
    dictionary.write()
    c = AlignableCorpus(flac_tg_corpus_dir, temp, use_mp=False)
    c.initialize_corpus(dictionary)
    c.files['61-70968-0000'].save()
