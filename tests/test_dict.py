import os

from montreal_forced_aligner.dictionary.multispeaker import MultispeakerDictionary
from montreal_forced_aligner.dictionary.pronunciation import PronunciationDictionary


def test_basic(basic_dict_path, generated_dir):
    output_directory = os.path.join(generated_dir, "dictionary_tests")
    dictionary = PronunciationDictionary(
        dictionary_path=basic_dict_path, temporary_directory=output_directory
    )
    dictionary.write()
    dictionary.write(write_disambiguation=True)

    assert dictionary
    assert len(dictionary) > 0
    assert set(dictionary.phones) == {"sil", "sp", "spn", "phonea", "phoneb", "phonec"}
    assert set(dictionary.kaldi_non_silence_phones) == {
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
    output_directory = os.path.join(generated_dir, "dictionary_tests")
    dictionary = MultispeakerDictionary(
        dictionary_path=extra_annotations_path, temporary_directory=output_directory
    )
    dictionary.dictionary_setup()
    dictionary.write_lexicon_information()
    d = dictionary.default_dictionary
    assert "{" in d.graphemes


def test_basic_noposition(basic_dict_path, generated_dir):
    output_directory = os.path.join(generated_dir, "dictionary_tests")
    dictionary = MultispeakerDictionary(
        dictionary_path=basic_dict_path,
        position_dependent_phones=False,
        temporary_directory=output_directory,
    )
    dictionary.dictionary_setup()
    dictionary.write_lexicon_information()
    d = dictionary.default_dictionary
    assert set(d.phones) == {"sil", "sp", "spn", "phonea", "phoneb", "phonec"}


def test_frclitics(frclitics_dict_path, generated_dir):
    output_directory = os.path.join(generated_dir, "dictionary_tests")
    dictionary = MultispeakerDictionary(
        dictionary_path=frclitics_dict_path,
        position_dependent_phones=False,
        temporary_directory=output_directory,
    )
    dictionary.dictionary_setup()
    dictionary.write_lexicon_information()
    d = dictionary.default_dictionary
    data = d.data()
    assert d.silences == data.silence_phones
    assert d.multilingual_ipa == data.multilingual_ipa
    assert d.words_mapping == data.words_mapping
    assert d.punctuation == data.punctuation
    assert d.clitic_markers == data.clitic_markers
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
    assert d.split_clitics("m'c'est") == ["m'", "c'", "est"]
    assert d.split_clitics("purple-people-eater") == ["purple-people-eater"]
    assert d.split_clitics("m'appele") == ["m'", "appele"]
    assert d.split_clitics("m'ving-sic") == ["m'", "ving", "sic"]
    assert d.split_clitics("flying'purple-people-eater") == ["flying'purple-people-eater"]

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
    assert d.to_int("m'c'est") == [
        d.words_mapping["m'"],
        d.words_mapping["c'"],
        d.words_mapping["est"],
    ]
    assert d.to_int("purple-people-eater") == [d.oov_int]
    assert d.to_int("m'appele") == [d.words_mapping["m'"], d.oov_int]
    assert d.to_int("m'ving-sic") == [d.words_mapping["m'"], d.oov_int, d.oov_int]
    assert d.to_int("flying'purple-people-eater") == [d.oov_int]


def test_english_clitics(english_dictionary, generated_dir):
    output_directory = os.path.join(generated_dir, "dictionary_tests")
    dictionary = MultispeakerDictionary(
        dictionary_path=english_dictionary,
        position_dependent_phones=False,
        temporary_directory=output_directory,
    )
    dictionary.dictionary_setup()
    dictionary.write_lexicon_information()
    d = dictionary.default_dictionary
    assert d.split_clitics("l'orme's") == ["l'", "orme's"]

    assert d.to_int("l'orme's") == [d.words_mapping["l'"], d.words_mapping["orme's"]]


def test_devanagari(english_dictionary, generated_dir):
    output_directory = os.path.join(generated_dir, "dictionary_tests")
    d = PronunciationDictionary(
        dictionary_path=english_dictionary,
        position_dependent_phones=False,
        temporary_directory=output_directory,
    )
    test_cases = ["हैं", "हूं", "हौं"]
    for tc in test_cases:
        assert tc == d.sanitize(tc)


def test_japanese(english_dictionary, generated_dir):
    output_directory = os.path.join(generated_dir, "dictionary_tests")
    d = PronunciationDictionary(
        dictionary_path=english_dictionary,
        position_dependent_phones=False,
        temporary_directory=output_directory,
    )
    assert "かぎ括弧" == d.sanitize("「かぎ括弧」")
    assert "二重かぎ括弧" == d.sanitize("『二重かぎ括弧』")


def test_multilingual_ipa(english_dictionary, generated_dir):
    output_directory = os.path.join(generated_dir, "dictionary_tests")
    dictionary = MultispeakerDictionary(
        dictionary_path=english_dictionary,
        position_dependent_phones=False,
        multilingual_ipa=True,
        temporary_directory=output_directory,
    )

    dictionary.dictionary_setup()
    dictionary.write_lexicon_information()
    d = dictionary.default_dictionary
    input_transcription = "m æ ŋ g oʊ dʒ aɪ".split()
    expected = tuple("m æ ŋ ɡ o ʊ d ʒ a ɪ".split())
    assert d.parse_ipa(input_transcription) == expected

    input_transcription = "n ɔː ɹ job_name".split()
    expected = tuple("n ɔ ɹ job_name".split())
    assert d.parse_ipa(input_transcription) == expected

    input_transcription = "t ʌ tʃ ə b l̩".split()
    expected = tuple("t ʌ t ʃ ə b l".split())
    assert d.parse_ipa(input_transcription) == expected


def test_xsampa_dir(xsampa_dict_path, generated_dir):
    output_directory = os.path.join(generated_dir, "dictionary_tests")

    dictionary = MultispeakerDictionary(
        dictionary_path=xsampa_dict_path,
        position_dependent_phones=False,
        multilingual_ipa=True,
        punctuation=list(".-']["),
        temporary_directory=output_directory,
    )

    dictionary.dictionary_setup()
    dictionary.write_lexicon_information()
    d = dictionary.default_dictionary
    print(d.words)
    assert not d.clitic_set
    assert d.split_clitics(r"r\{und") == [r"r\{und"]
    assert d.split_clitics("{bI5s@`n") == ["{bI5s@`n"]
    assert d.words[r"r\{und"]


def test_multispeaker_config(multispeaker_dictionary_config_path, sick_corpus, generated_dir):
    output_directory = os.path.join(generated_dir, "dictionary_tests")
    dictionary = MultispeakerDictionary(
        dictionary_path=multispeaker_dictionary_config_path,
        position_dependent_phones=False,
        multilingual_ipa=True,
        punctuation=list(".-']["),
        temporary_directory=output_directory,
    )
    dictionary.dictionary_setup()
    dictionary.write_lexicon_information()
    for d in dictionary.dictionary_mapping.values():
        assert d.silence_phones.issubset(dictionary.silence_phones)
        assert d.non_silence_phones.issubset(dictionary.non_silence_phones)
