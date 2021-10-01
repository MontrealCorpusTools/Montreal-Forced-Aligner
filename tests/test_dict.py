import os
import pytest

from montreal_forced_aligner.dictionary import Dictionary, MultispeakerDictionary, sanitize, parse_ipa
from montreal_forced_aligner.textgrid import split_clitics


def ListLines(path):
    lines = []
    thefile = open(path)
    text = thefile.readlines()
    for line in text:
        stripped = line.strip()
        if stripped != '':
            lines.append(stripped)
    return lines


def test_basic(basic_dict_path, generated_dir):
    d = Dictionary(basic_dict_path, os.path.join(generated_dir, 'basic'))
    d.write()
    assert set(d.phones) == {'sil', 'sp', 'spn', 'phonea', 'phoneb', 'phonec'}
    assert set(d.positional_nonsil_phones) == {'phonea_B', 'phonea_I', 'phonea_E', 'phonea_S',
                                               'phoneb_B', 'phoneb_I', 'phoneb_E', 'phoneb_S',
                                               'phonec_B', 'phonec_I', 'phonec_E', 'phonec_S'}


def test_extra_annotations(extra_annotations_path, generated_dir):
    d = Dictionary(extra_annotations_path, os.path.join(generated_dir, 'extra'))
    assert '{' in d.graphemes
    d.write()


def test_basic_noposition(basic_dict_path, generated_dir):
    d = Dictionary(basic_dict_path, os.path.join(generated_dir, 'basic'), position_dependent_phones=False)
    d.write()
    assert set(d.phones) == {'sil', 'sp', 'spn', 'phonea', 'phoneb', 'phonec'}


def test_frclitics(frclitics_dict_path, generated_dir):
    d = Dictionary(frclitics_dict_path, os.path.join(generated_dir, 'frclitics'))
    d.write()
    assert not d.check_word('aujourd')
    assert d.check_word('aujourd\'hui')
    assert d.check_word('m\'appelle')
    assert not d.check_word('purple-people-eater')
    assert d.split_clitics('aujourd') == ['aujourd']
    assert split_clitics('aujourd', d.words_mapping, d.clitic_set, d.clitic_markers, d.compound_markers) == ['aujourd']
    assert d.split_clitics('aujourd\'hui') == ['aujourd\'hui']
    assert split_clitics('aujourd\'hui', d.words_mapping, d.clitic_set, d.clitic_markers, d.compound_markers)  == ['aujourd\'hui']
    assert d.split_clitics('vingt-six') == ['vingt', 'six']
    assert split_clitics('vingt-six', d.words_mapping, d.clitic_set, d.clitic_markers, d.compound_markers) == ['vingt', 'six']
    assert d.split_clitics('m\'appelle') == ['m\'', 'appelle']
    assert split_clitics('m\'appelle', d.words_mapping, d.clitic_set, d.clitic_markers, d.compound_markers) == ['m\'', 'appelle']
    assert d.split_clitics('m\'m\'appelle') == ['m\'', 'm\'', 'appelle']
    assert split_clitics('m\'m\'appelle', d.words_mapping, d.clitic_set, d.clitic_markers, d.compound_markers) == ['m\'', 'm\'', 'appelle']
    assert d.split_clitics('c\'est') == ['c\'est']
    assert split_clitics('c\'est', d.words_mapping, d.clitic_set, d.clitic_markers, d.compound_markers) == ['c\'est']
    assert d.split_clitics('m\'c\'est') == ['m\'', 'c\'est']
    assert split_clitics('m\'c\'est', d.words_mapping, d.clitic_set, d.clitic_markers, d.compound_markers) == ['m\'', 'c\'est']
    assert d.split_clitics('purple-people-eater') == ['purple', 'people', 'eater']
    assert split_clitics('purple-people-eater', d.words_mapping, d.clitic_set, d.clitic_markers, d.compound_markers) == ['purple', 'people', 'eater']
    assert d.split_clitics('m\'appele') == ['m\'', 'appele']
    assert split_clitics('m\'appele', d.words_mapping, d.clitic_set, d.clitic_markers, d.compound_markers) == ['m\'', 'appele']
    assert d.split_clitics('m\'ving-sic') == ["m'", 'ving', 'sic']
    assert split_clitics('m\'ving-sic', d.words_mapping, d.clitic_set, d.clitic_markers, d.compound_markers) == ["m'", 'ving', 'sic']
    assert d.split_clitics('flying\'purple-people-eater') == ['flying\'purple', 'people', 'eater']
    assert split_clitics('flying\'purple-people-eater', d.words_mapping, d.clitic_set, d.clitic_markers, d.compound_markers) == ['flying\'purple', 'people', 'eater']

    assert d.to_int('aujourd') == [d.oov_int]
    assert d.to_int('aujourd\'hui') == [d.words_mapping['aujourd\'hui']]
    assert d.to_int('vingt-six') == [d.words_mapping['vingt'], d.words_mapping['six']]
    assert d.to_int('m\'appelle') == [d.words_mapping['m\''], d.words_mapping['appelle']]
    assert d.to_int('m\'m\'appelle') == [d.words_mapping['m\''], d.words_mapping['m\''], d.words_mapping['appelle']]
    assert d.to_int('c\'est') == [d.words_mapping['c\'est']]
    assert d.to_int('m\'c\'est') == [d.words_mapping['m\''], d.words_mapping['c\'est']]
    assert d.to_int('purple-people-eater') == [d.oov_int]
    assert d.to_int('m\'appele') == [d.words_mapping['m\''], d.oov_int]
    assert d.to_int('m\'ving-sic') == [d.words_mapping['m\''], d.oov_int, d.oov_int]
    assert d.to_int('flying\'purple-people-eater') == [d.oov_int]


def test_english_clitics(english_pretrained_dictionary, generated_dir):
    d = Dictionary(english_pretrained_dictionary, os.path.join(generated_dir, 'english_clitic_test'))
    d.write()
    assert d.split_clitics("l'orme's") == ["l'", "orme's"]

    assert d.to_int("l'orme's") == [d.words_mapping["l'"], d.words_mapping["orme's"]]


def test_devanagari():
    test_cases = ["हैं", "हूं", "हौं"]
    for tc in test_cases:
        assert tc == sanitize(tc)


def test_japanese():
    assert "かぎ括弧" == sanitize("「かぎ括弧」")
    assert "二重かぎ括弧" == sanitize("『二重かぎ括弧』")


def test_multilingual_ipa():
    input_transcription = 'm æ ŋ g oʊ dʒ aɪ'.split()
    expected = tuple('m æ ŋ ɡ o ʊ d ʒ a ɪ'.split())
    assert parse_ipa(input_transcription) == expected

    input_transcription = 'n ɔː ɹ i'.split()
    expected = tuple('n ɔ ɹ i'.split())
    assert parse_ipa(input_transcription) == expected

    input_transcription = 't ʌ tʃ ə b l̩'.split()
    expected = tuple('t ʌ t ʃ ə b l'.split())
    assert parse_ipa(input_transcription) == expected


def test_multispeaker_config(multispeaker_dictionary_config, generated_dir):
    dictionary = MultispeakerDictionary(multispeaker_dictionary_config, os.path.join(generated_dir, 'multispeaker'))
    dictionary.write()
    for name, d in dictionary.dictionary_mapping.items():
        assert d.sil_phones.issubset(dictionary.sil_phones)
        assert d.nonsil_phones.issubset(dictionary.nonsil_phones)
        assert set(d.words.keys()).issubset(dictionary.words)