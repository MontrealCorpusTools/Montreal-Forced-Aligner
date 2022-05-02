import os
import shutil

import pytest

from montreal_forced_aligner.alignment.pretrained import PretrainedAligner
from montreal_forced_aligner.dictionary.multispeaker import MultispeakerDictionary


def test_abstract(abstract_dict_path, generated_dir):
    output_directory = os.path.join(generated_dir, "dictionary_tests", "abstract")
    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = MultispeakerDictionary(
        dictionary_path=abstract_dict_path, temporary_directory=output_directory
    )
    dictionary.dictionary_setup()

    assert dictionary
    assert set(dictionary.phones) == {"sil", "spn", "phonea", "phoneb", "phonec"}
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


def test_tabbed(tabbed_dict_path, basic_dict_path, generated_dir):
    output_directory = os.path.join(generated_dir, "dictionary_tests", "tabbed")
    shutil.rmtree(output_directory, ignore_errors=True)
    tabbed_dictionary = MultispeakerDictionary(
        dictionary_path=tabbed_dict_path, temporary_directory=output_directory
    )
    tabbed_dictionary.dictionary_setup()
    basic_dictionary = MultispeakerDictionary(
        dictionary_path=basic_dict_path, temporary_directory=output_directory
    )
    basic_dictionary.dictionary_setup()
    assert tabbed_dictionary.word_mapping(1) == basic_dictionary.word_mapping(1)


@pytest.mark.skip("Outdated models")
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
    output_directory = os.path.join(generated_dir, "dictionary_tests", "extras")
    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = MultispeakerDictionary(
        dictionary_path=extra_annotations_path, temporary_directory=output_directory
    )
    graphemes, _ = dictionary.dictionary_setup()
    dictionary.write_lexicon_information()
    assert "{" in graphemes


def test_abstract_noposition(abstract_dict_path, generated_dir):
    output_directory = os.path.join(generated_dir, "dictionary_tests", "abstract_no_position")
    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = MultispeakerDictionary(
        dictionary_path=abstract_dict_path,
        position_dependent_phones=False,
        temporary_directory=output_directory,
    )
    graphemes, _ = dictionary.dictionary_setup()
    dictionary.write_lexicon_information()
    assert set(dictionary.phones) == {"sil", "spn", "phonea", "phoneb", "phonec"}


def test_frclitics(frclitics_dict_path, generated_dir):
    output_directory = os.path.join(generated_dir, "dictionary_tests", "fr_clitics")
    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = MultispeakerDictionary(
        dictionary_path=frclitics_dict_path,
        position_dependent_phones=False,
        temporary_directory=output_directory,
    )
    dictionary.dictionary_setup()
    dictionary.write_lexicon_information()
    s, spl = dictionary.sanitize_function.get_functions_for_speaker("default")
    assert spl.to_int("aujourd") == spl.word_mapping[spl.oov_word]
    assert spl.to_int("aujourd'hui") != spl.word_mapping[spl.oov_word]
    assert spl.to_int("m'appelle") == spl.word_mapping[spl.oov_word]
    assert spl.to_int("purple-people-eater") == spl.word_mapping[spl.oov_word]
    assert spl("aujourd") == ["aujourd"]
    assert spl("aujourd'hui") == ["aujourd'hui"]
    assert spl("vingt-six") == ["vingt", "six"]
    assert spl("m'appelle") == ["m'", "appelle"]
    assert spl("m'm'appelle") == ["m'", "m'", "appelle"]
    assert spl("c'est") == ["c'est"]
    assert spl("m'c'est") == ["m'", "c'", "est"]
    assert spl("purple-people-eater") == ["purple-people-eater"]
    assert spl("m'appele") == ["m'", "appele"]
    assert spl("m'ving-sic") == ["m'", "ving", "sic"]
    assert spl("flying'purple-people-eater") == ["flying'purple-people-eater"]


def test_english_clitics(english_dictionary, generated_dir):
    output_directory = os.path.join(generated_dir, "dictionary_tests", "english_clitics")
    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = MultispeakerDictionary(
        dictionary_path=english_dictionary,
        position_dependent_phones=False,
        temporary_directory=output_directory,
        phone_set_type="AUTO",
    )
    dictionary.dictionary_setup()
    dictionary.write_lexicon_information()
    assert dictionary.phone_set_type.name == "ARPA"
    assert dictionary.extra_questions_mapping
    for k, v in dictionary.extra_questions_mapping.items():
        print(k)
        print(v)
        assert len(v) == len(set(v))
    assert all(x.endswith("0") for x in dictionary.extra_questions_mapping["stress_0"])
    assert all(x.endswith("1") for x in dictionary.extra_questions_mapping["stress_1"])
    assert all(x.endswith("2") for x in dictionary.extra_questions_mapping["stress_2"])
    assert "fricatives" in dictionary.extra_questions_mapping
    voiceless_fricatives = {
        "V",
        "DH",
        "HH",
        "F",
        "TH",
    }
    assert all(x in dictionary.extra_questions_mapping["fricatives"] for x in voiceless_fricatives)
    assert set(dictionary.extra_questions_mapping["close"]) == {"IH", "UH", "IY", "UW"}
    assert set(dictionary.extra_questions_mapping["close_mid"]) == {"EY", "OW", "AH"}

    s, spl = dictionary.sanitize_function.get_functions_for_speaker("default")
    assert spl.split_clitics("l'orme's") == ["l'", "orme", "'s"]


def test_english_mfa(english_us_mfa_dictionary, generated_dir):
    output_directory = os.path.join(generated_dir, "dictionary_tests", "english_mfa")
    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = MultispeakerDictionary(
        dictionary_path=english_us_mfa_dictionary,
        position_dependent_phones=False,
        temporary_directory=output_directory,
        phone_set_type="AUTO",
    )
    dictionary.dictionary_setup()
    dictionary.write_lexicon_information()
    assert dictionary.phone_set_type.name == "IPA"
    assert dictionary.extra_questions_mapping
    for k, v in dictionary.extra_questions_mapping.items():
        print(k)
        print(v)
        assert len(v) == len(set(v))
    assert "dental" in dictionary.extra_questions_mapping
    dental = {"f", "v", "θ", "ð"}
    assert all(x in dictionary.extra_questions_mapping["dental"] for x in dental)
    # assert set(d.extra_questions_mapping["central"]) == {'ʉ', 'ə', 'ɐ', 'ɝ', 'ɚ', 'ɝː'}
    # assert set(d.extra_questions_mapping["close"]) == {'ɪ', 'ʉ', 'ʊ', 'i'}


def test_mandarin_pinyin(pinyin_dictionary, generated_dir):
    output_directory = os.path.join(generated_dir, "dictionary_tests", "pinyin")
    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = MultispeakerDictionary(
        dictionary_path=pinyin_dictionary,
        position_dependent_phones=False,
        temporary_directory=output_directory,
        phone_set_type="AUTO",
    )
    dictionary.dictionary_setup()
    dictionary.write_lexicon_information()
    assert dictionary.phone_set_type.name == "PINYIN"
    assert dictionary.extra_questions_mapping
    for k, v in dictionary.extra_questions_mapping.items():
        print(k)
        print(v)
        assert len(v) == len(set(v))
    assert "voiceless_sibilant_variation" in dictionary.extra_questions_mapping
    voiceless_fricatives = ["z", "zh", "j", "c", "ch", "q", "s", "sh", "x"]
    assert all(
        x in dictionary.extra_questions_mapping["voiceless_sibilant_variation"]
        for x in voiceless_fricatives
    )
    assert set(dictionary.extra_questions_mapping["rhotic_variation"]) == {
        "e5",
        "e1",
        "sh",
        "e4",
        "e2",
        "r",
        "e3",
    }
    assert set(dictionary.extra_questions_mapping["dorsal_variation"]) == {"h", "k", "g"}
    assert "uai1" in dictionary.extra_questions_mapping["tone_1"]


def test_devanagari(english_dictionary, generated_dir):
    output_directory = os.path.join(generated_dir, "dictionary_tests", "devanagari")
    shutil.rmtree(output_directory, ignore_errors=True)
    d = MultispeakerDictionary(
        dictionary_path=english_dictionary,
        position_dependent_phones=False,
        temporary_directory=output_directory,
    )
    test_cases = ["हैं", "हूं", "हौं"]
    for tc in test_cases:
        assert [tc] == list(d.sanitize(tc))


def test_japanese(english_dictionary, generated_dir):
    output_directory = os.path.join(generated_dir, "dictionary_tests", "japanese")
    shutil.rmtree(output_directory, ignore_errors=True)
    d = MultispeakerDictionary(
        dictionary_path=english_dictionary,
        position_dependent_phones=False,
        temporary_directory=output_directory,
    )
    assert ["かぎ括弧"] == list(d.sanitize("「かぎ括弧」"))
    assert ["二重かぎ括弧"] == list(d.sanitize("『二重かぎ括弧』"))


def test_xsampa_dir(xsampa_dict_path, generated_dir):
    output_directory = os.path.join(generated_dir, "dictionary_tests", "xsampa")
    shutil.rmtree(output_directory, ignore_errors=True)

    dictionary = MultispeakerDictionary(
        dictionary_path=xsampa_dict_path,
        position_dependent_phones=False,
        punctuation=list(".-']["),
        temporary_directory=output_directory,
    )

    dictionary.dictionary_setup()
    dictionary.write_lexicon_information()
    s, spl = dictionary.sanitize_function.get_functions_for_speaker("default")
    assert not spl.clitic_set
    assert spl.split_clitics(r"r\{und") == [r"r\{und"]
    assert spl.split_clitics("{bI5s@`n") == ["{bI5s@`n"]
    assert dictionary.word_mapping(1)[r"r\{und"]


def test_multispeaker_config(multispeaker_dictionary_config_path, generated_dir):
    output_directory = os.path.join(generated_dir, "dictionary_tests", "multispeaker")
    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = MultispeakerDictionary(
        dictionary_path=multispeaker_dictionary_config_path,
        position_dependent_phones=False,
        punctuation=list(".-']["),
        temporary_directory=output_directory,
    )
    dictionary.dictionary_setup()
    dictionary.write_lexicon_information()


def test_vietnamese_tones(vietnamese_dict_path, generated_dir):
    output_directory = os.path.join(generated_dir, "dictionary_tests", "vietnamese")
    shutil.rmtree(output_directory, ignore_errors=True)
    d = MultispeakerDictionary(
        dictionary_path=vietnamese_dict_path,
        position_dependent_phones=False,
        temporary_directory=output_directory,
        phone_set_type="IPA",
    )
    graphemes, phone_counts = d.dictionary_setup()
    assert d.get_base_phone("o˨˩ˀ") == "o"
    assert "o˦˩" in phone_counts
    assert "o" in d.kaldi_grouped_phones
    assert "o˨˩ˀ" in d.kaldi_grouped_phones["o"]
    assert "o˦˩" in d.kaldi_grouped_phones["o"]
    d.db_engine.dispose()

    output_directory = os.path.join(generated_dir, "dictionary_tests", "vietnamese_keep_tone")
    d = MultispeakerDictionary(
        dictionary_path=vietnamese_dict_path,
        position_dependent_phones=False,
        temporary_directory=output_directory,
        preserve_suprasegmentals=True,
        phone_set_type="IPA",
    )
    graphemes, phone_counts = d.dictionary_setup()

    assert d.get_base_phone("o˨˩ˀ") == "o˨˩ˀ"
    assert "o" not in d.kaldi_grouped_phones
    assert "o˨˩ˀ" in d.kaldi_grouped_phones
    assert "o˦˩" in d.kaldi_grouped_phones
