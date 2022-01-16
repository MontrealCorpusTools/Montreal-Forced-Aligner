import os

import pytest

from montreal_forced_aligner.alignment.pretrained import PretrainedAligner
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
    assert set(dictionary.phones) == {"sil", "noi", "spn", "phonea", "phoneb", "phonec"}
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


def test_missing_phones(
    basic_corpus_dir, generated_dir, german_prosodylab_acoustic_model, german_prosodylab_dictionary
):

    output_directory = os.path.join(generated_dir, "dictionary_tests")
    aligner = PretrainedAligner(
        acoustic_model_path=german_prosodylab_acoustic_model,
        corpus_directory=basic_corpus_dir,
        dictionary_path=german_prosodylab_dictionary,
        temporary_directory=output_directory,
    )
    aligner.setup()


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
    assert set(d.phones) == {"sil", "noi", "spn", "phonea", "phoneb", "phonec"}


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
        phone_set_type="AUTO",
    )
    dictionary.dictionary_setup()
    dictionary.write_lexicon_information()
    d = dictionary.default_dictionary
    assert dictionary.phone_set_type.name == "ARPA"
    assert d.extra_questions_mapping
    assert d.phone_set_type.name == "ARPA"
    for k, v in d.extra_questions_mapping.items():
        print(k)
        print(v)
        assert len(v) == len(set(v))
    assert all(x.endswith("0") for x in d.extra_questions_mapping["stress_0"])
    assert all(x.endswith("1") for x in d.extra_questions_mapping["stress_1"])
    assert all(x.endswith("2") for x in d.extra_questions_mapping["stress_2"])
    assert "voiceless_fricative_variation" in d.extra_questions_mapping
    voiceless_fricatives = ["F", "HH", "K", "TH"]
    assert all(
        x in d.extra_questions_mapping["voiceless_fricative_variation"]
        for x in voiceless_fricatives
    )
    assert set(d.extra_questions_mapping["high_back_variation"]) == {
        "UH0",
        "UH1",
        "UH2",
        "UW0",
        "UW1",
        "UW2",
    }
    assert set(d.extra_questions_mapping["central_variation"]) == {
        "ER0",
        "ER1",
        "ER2",
        "AH0",
        "AH1",
        "AH2",
        "UH0",
        "UH1",
        "UH2",
        "IH0",
        "IH1",
        "IH2",
    }

    topos = d.kaldi_phones_for_topo
    print(topos)
    assert 1 in topos
    assert 2 in topos
    assert "AY1" in topos[5]
    assert "JH" in topos[4]
    assert "B" in topos[2]
    assert "NG" in topos[3]
    assert set(topos[1]) == {"AH0", "ER0", "UH0", "IH0"}
    assert d.split_clitics("l'orme's") == ["l'", "orme", "'s"]

    assert d.to_int("l'orme's") == [
        d.words_mapping["l'"],
        d.words_mapping["orme"],
        d.words_mapping["'s"],
    ]


def test_english_ipa(english_us_ipa_dictionary, generated_dir):
    output_directory = os.path.join(generated_dir, "dictionary_tests")
    dictionary = MultispeakerDictionary(
        dictionary_path=english_us_ipa_dictionary,
        position_dependent_phones=False,
        temporary_directory=output_directory,
        phone_set_type="AUTO",
    )
    dictionary.dictionary_setup()
    dictionary.write_lexicon_information()
    d = dictionary.default_dictionary
    assert dictionary.phone_set_type.name == "IPA"
    assert d.extra_questions_mapping
    assert d.phone_set_type.name == "IPA"
    for k, v in d.extra_questions_mapping.items():
        print(k)
        print(v)
        assert len(v) == len(set(v))
    assert "voiceless_fricative_variation" in d.extra_questions_mapping
    voiceless_fricatives = [
        "θ",
        "f",
        "h",
    ]
    assert all(
        x in d.extra_questions_mapping["voiceless_fricative_variation"]
        for x in voiceless_fricatives
    )
    assert set(d.extra_questions_mapping["high_back_variation"]) == {"ʊ", "u", "uː"}
    assert set(d.extra_questions_mapping["central_variation"]) == {
        "ə",
        "ɚ",
        "ʌ",
        "ʊ",
        "ɝ",
        "ɝː",
    }

    topos = d.kaldi_phones_for_topo
    print(topos)
    assert 1 in topos
    assert 2 in topos
    assert "aɪ" in topos[5]
    assert "dʒ" in topos[4]
    assert "b" in topos[2]
    assert "ŋ" in topos[3]
    assert set(topos[1]) == {"ə", "ɚ", "ɾ"}


def test_mandarin_pinyin(pinyin_dictionary, generated_dir):
    output_directory = os.path.join(generated_dir, "dictionary_tests")
    dictionary = MultispeakerDictionary(
        dictionary_path=pinyin_dictionary,
        position_dependent_phones=False,
        temporary_directory=output_directory,
        phone_set_type="AUTO",
    )
    dictionary.dictionary_setup()
    dictionary.write_lexicon_information()
    d = dictionary.default_dictionary
    assert dictionary.phone_set_type.name == "PINYIN"
    assert d.extra_questions_mapping
    assert d.phone_set_type.name == "PINYIN"
    for k, v in d.extra_questions_mapping.items():
        print(k)
        print(v)
        assert len(v) == len(set(v))
    assert "voiceless_sibilant_variation" in d.extra_questions_mapping
    voiceless_fricatives = ["z", "zh", "j", "c", "ch", "q", "s", "sh", "x"]
    assert all(
        x in d.extra_questions_mapping["voiceless_sibilant_variation"]
        for x in voiceless_fricatives
    )
    assert set(d.extra_questions_mapping["rhotic_variation"]) == {
        "e5",
        "e1",
        "sh",
        "e4",
        "e2",
        "r",
        "e3",
    }
    assert set(d.extra_questions_mapping["dorsal_variation"]) == {"h", "k", "g"}
    assert "uai1" in d.extra_questions_mapping["tone_1"]

    topos = d.kaldi_phones_for_topo
    print(topos)
    assert 2 in topos
    assert 5 in topos
    assert "ai" in topos[5]
    assert "ch" in topos[5]
    assert "z" in topos[4]
    assert "b" in topos[2]
    assert "p" in topos[3]


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


@pytest.mark.skip
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
        punctuation=list(".-']["),
        temporary_directory=output_directory,
    )
    dictionary.dictionary_setup()
    dictionary.write_lexicon_information()
    for d in dictionary.dictionary_mapping.values():
        assert d.silence_phones.issubset(dictionary.silence_phones)
        assert d.non_silence_phones.issubset(dictionary.non_silence_phones)
