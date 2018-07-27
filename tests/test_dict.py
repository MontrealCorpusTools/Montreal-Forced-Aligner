import os
import pytest

from aligner.dictionary import Dictionary


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
    x = d.write()
    assert set(d.phones) == {'sil', 'sp', 'spn', 'phonea', 'phoneb', 'phonec'}


def test_frclitics(frclitics_dict_path, generated_dir):
    d = Dictionary(frclitics_dict_path, os.path.join(generated_dir, 'frclitics'))
    x = d.write()
    assert d.separate_clitics('aujourd') == ['aujourd']
    assert d.separate_clitics('aujourd\'hui') == ['aujourd\'hui']
    assert d.separate_clitics('vingt-six') == ['vingt', 'six']
    assert d.separate_clitics('m\'appelle') == ['m\'', 'appelle']
    assert d.separate_clitics('c\'est') == ['c\'est']
    assert d.separate_clitics('purple-people-eater') == ['purple-people-eater']
    assert d.separate_clitics('m\'appele') == ['m\'', 'appele']
    assert d.separate_clitics('m\'ving-sic') == ["m'", 'ving', 'sic']
    assert d.separate_clitics('flying\'purple-people-eater') == ['flying\'purple-people-eater']
