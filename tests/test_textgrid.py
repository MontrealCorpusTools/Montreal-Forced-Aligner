import os
import pytest

from montreal_forced_aligner.config.base_config import DEFAULT_STRIP_DIACRITICS, DEFAULT_DIGRAPHS
from montreal_forced_aligner.textgrid import map_to_original_pronunciation


def test_mapping():
    cur_phones = [[2.25, 2.33, 't'], [2.33, 2.43, 'ʃ'], [2.43, 2.55, 'æ'], [2.55, 2.64, 'd'], [2.64, 2.71, 'l'], [2.71, 2.78, 'a'], [2.78, 2.84, 'ɪ'], [2.84, 2.92, 'k']]
    subprons = [[{'pronunciation': ('t', 'ʃ', 'æ', 'd'), 'probability': None, 'disambiguation': None, 'right_sil_prob': None, 'left_sil_prob': None, 'left_nonsil_prob': None, 'original_pronunciation': ('tʃ', 'æ', 'd')}],
                [{'pronunciation': ('l', 'a', 'ɪ', 'k'), 'probability': None, 'disambiguation': None, 'right_sil_prob': None, 'left_sil_prob': None, 'left_nonsil_prob': None, 'original_pronunciation': ('l', 'aɪ', 'k')}]]
    new_phones = map_to_original_pronunciation(cur_phones, subprons, DEFAULT_STRIP_DIACRITICS)
    assert new_phones == [[2.25, 2.43, 'tʃ'], [2.43, 2.55, 'æ'], [2.55, 2.64, 'd'], [2.64, 2.71, 'l'], [2.71, 2.84, 'aɪ'], [2.84, 2.92, 'k']]