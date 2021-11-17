import os

from montreal_forced_aligner.corpus import Corpus
from montreal_forced_aligner.dictionary import MultispeakerDictionary


def test_save_text_lab(
    basic_dict_path,
    basic_corpus_dir,
    generated_dir,
    default_feature_config,
    basic_dictionary_config,
):
    dictionary = MultispeakerDictionary(
        basic_dict_path, os.path.join(generated_dir, "basic"), basic_dictionary_config
    )
    dictionary.write()
    output_directory = os.path.join(generated_dir, "basic")
    c = Corpus(basic_corpus_dir, output_directory, basic_dictionary_config, use_mp=True)
    c.initialize_corpus(dictionary)
    c.files["acoustic_corpus"].save()


def test_flac_tg(
    basic_dict_path, flac_tg_corpus_dir, temp_dir, default_feature_config, basic_dictionary_config
):
    temp = os.path.join(temp_dir, "flac_tg_corpus")
    dictionary = MultispeakerDictionary(
        basic_dict_path, os.path.join(temp, "basic"), basic_dictionary_config
    )
    dictionary.write()
    c = Corpus(flac_tg_corpus_dir, temp, basic_dictionary_config, use_mp=False)
    c.initialize_corpus(dictionary)
    c.files["61-70968-0000"].save()
