import os

from montreal_forced_aligner.config.dictionary_config import DictionaryConfig
from montreal_forced_aligner.config.train_config import train_yaml_to_config
from montreal_forced_aligner.dictionary import MultispeakerDictionary, PronunciationDictionary


def ListLines(path):
    lines = []
    thefile = open(path)
    text = thefile.readlines()
    for line in text:
        stripped = line.strip()
        if stripped != "":
            lines.append(stripped)
    return lines


def test_basic(basic_dict_path, generated_dir):
    d = PronunciationDictionary(basic_dict_path, os.path.join(generated_dir, "basic"))
    d.write()
    assert set(d.config.phones) == {"sil", "sp", "spn", "phonea", "phoneb", "phonec"}
    assert set(d.config.kaldi_non_silence_phones) == {
        "phonea_B",
        "phonea_I",
        "phonea_E",
        "phonea_S",
        "phoneb_B",
        "phoneb_I",
        "phoneb_E",
        "phoneb_S",
        "phonec_B",
        "phonec_I",
        "phonec_E",
        "phonec_S",
    }


def test_extra_annotations(extra_annotations_path, generated_dir):
    d = PronunciationDictionary(extra_annotations_path, os.path.join(generated_dir, "extra"))
    assert "{" in d.graphemes
    d.write()


def test_basic_noposition(basic_dict_path, generated_dir):
    config = DictionaryConfig(position_dependent_phones=False)
    d = PronunciationDictionary(basic_dict_path, os.path.join(generated_dir, "basic"), config)
    d.write()
    assert set(d.config.phones) == {"sil", "sp", "spn", "phonea", "phoneb", "phonec"}


def test_frclitics(frclitics_dict_path, generated_dir):
    d = PronunciationDictionary(frclitics_dict_path, os.path.join(generated_dir, "frclitics"))
    d.write()
    data = d.data()
    assert d.silences == data.dictionary_config.silence_phones
    assert d.config.multilingual_ipa == data.dictionary_config.multilingual_ipa
    assert d.words_mapping == data.words_mapping
    assert d.config.punctuation == data.dictionary_config.punctuation
    assert d.config.clitic_markers == data.dictionary_config.clitic_markers
    assert d.oov_int == data.oov_int
    assert d.words == data.words
    assert not d.check_word("aujourd")
    assert d.check_word("aujourd'hui")
    assert d.check_word("m'appelle")
    assert not d.check_word("purple-people-eater")
    assert d.split_clitics("aujourd") == ["aujourd"]
    assert d.split_clitics("aujourd'hui") == ["aujourd'hui"]
    assert d.split_clitics("vingt-six") == ["vingt", "six"]
    assert d.split_clitics("m'appelle") == ["m'", "appelle"]
    assert d.split_clitics("m'm'appelle") == ["m'", "m'", "appelle"]
    assert d.split_clitics("c'est") == ["c'est"]
    assert d.split_clitics("m'c'est") == ["m'", "c'est"]
    assert d.split_clitics("purple-people-eater") == ["purple", "people", "eater"]
    assert d.split_clitics("m'appele") == ["m'", "appele"]
    assert d.split_clitics("m'ving-sic") == ["m'", "ving", "sic"]
    assert d.split_clitics("flying'purple-people-eater") == ["flying'purple", "people", "eater"]

    assert d.to_int("aujourd") == [d.oov_int]
    assert d.to_int("aujourd'hui") == [d.words_mapping["aujourd'hui"]]
    assert d.to_int("vingt-six") == [d.words_mapping["vingt"], d.words_mapping["six"]]
    assert d.to_int("m'appelle") == [d.words_mapping["m'"], d.words_mapping["appelle"]]
    assert d.to_int("m'm'appelle") == [
        d.words_mapping["m'"],
        d.words_mapping["m'"],
        d.words_mapping["appelle"],
    ]
    assert d.to_int("c'est") == [d.words_mapping["c'est"]]
    assert d.to_int("m'c'est") == [d.words_mapping["m'"], d.words_mapping["c'est"]]
    assert d.to_int("purple-people-eater") == [d.oov_int]
    assert d.to_int("m'appele") == [d.words_mapping["m'"], d.oov_int]
    assert d.to_int("m'ving-sic") == [d.words_mapping["m'"], d.oov_int, d.oov_int]
    assert d.to_int("flying'purple-people-eater") == [d.oov_int]


def test_english_clitics(english_dictionary, generated_dir, basic_dictionary_config):
    d = PronunciationDictionary(
        english_dictionary,
        os.path.join(generated_dir, "english_clitic_test"),
        basic_dictionary_config,
    )
    d.write()
    assert d.split_clitics("l'orme's") == ["l'", "orme's"]

    assert d.to_int("l'orme's") == [d.words_mapping["l'"], d.words_mapping["orme's"]]


def test_devanagari(basic_dictionary_config):
    test_cases = ["हैं", "हूं", "हौं"]
    for tc in test_cases:
        assert tc == basic_dictionary_config.sanitize(tc)


def test_japanese(basic_dictionary_config):
    assert "かぎ括弧" == basic_dictionary_config.sanitize("「かぎ括弧」")
    assert "二重かぎ括弧" == basic_dictionary_config.sanitize("『二重かぎ括弧』")


def test_multilingual_ipa(basic_dictionary_config):
    input_transcription = "m æ ŋ g oʊ dʒ aɪ".split()
    expected = tuple("m æ ŋ ɡ o ʊ d ʒ a ɪ".split())
    assert basic_dictionary_config.parse_ipa(input_transcription) == expected

    input_transcription = "n ɔː ɹ job_name".split()
    expected = tuple("n ɔ ɹ job_name".split())
    assert basic_dictionary_config.parse_ipa(input_transcription) == expected

    input_transcription = "t ʌ tʃ ə b l̩".split()
    expected = tuple("t ʌ t ʃ ə b l".split())
    assert basic_dictionary_config.parse_ipa(input_transcription) == expected


def test_xsampa_dir(xsampa_dict_path, generated_dir, different_punctuation_config):

    train_config, align_config, dictionary_config = train_yaml_to_config(
        different_punctuation_config
    )
    d = PronunciationDictionary(
        xsampa_dict_path, os.path.join(generated_dir, "xsampa"), dictionary_config
    )
    d.write()

    print(d.words)
    assert not d.config.clitic_set
    assert d.split_clitics(r"r\{und") == [r"r\{und"]
    assert d.split_clitics("{bI5s@`n") == ["{bI5s@`n"]
    assert d.words[r"r\{und"]


def test_multispeaker_config(
    multispeaker_dictionary_config, sick_corpus, basic_dictionary_config, generated_dir
):
    dictionary = MultispeakerDictionary(
        multispeaker_dictionary_config,
        os.path.join(generated_dir, "multispeaker"),
        basic_dictionary_config,
        word_set=sick_corpus.word_set,
    )
    dictionary.write()
    for d in dictionary.dictionary_mapping.values():
        assert d.silences.issubset(dictionary.config.silence_phones)
        assert d.config.non_silence_phones.issubset(dictionary.config.non_silence_phones)
