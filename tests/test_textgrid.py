import os

import pytest

from montreal_forced_aligner.dictionary.multispeaker import MultispeakerDictionary
from montreal_forced_aligner.textgrid import CtmInterval


@pytest.mark.skip
def test_mapping(english_us_ipa_dictionary, generated_dir):
    output_directory = os.path.join(generated_dir, "textgrid_tests")
    dictionary = MultispeakerDictionary(
        dictionary_path=english_us_ipa_dictionary,
        position_dependent_phones=False,
        multilingual_ipa=True,
        temporary_directory=output_directory,
    )
    dictionary.dictionary_setup()
    dictionary.write_lexicon_information()
    d = dictionary.default_dictionary
    u = "utt"
    cur_phones = [
        CtmInterval(2.25, 2.33, "t", u),
        CtmInterval(2.33, 2.43, "ʃ", u),
        CtmInterval(2.43, 2.55, "æ", u),
        CtmInterval(2.55, 2.64, "d", u),
        CtmInterval(2.64, 2.71, "l", u),
        CtmInterval(2.71, 2.78, "a", u),
        CtmInterval(2.78, 2.84, "ɪ", u),
        CtmInterval(2.84, 2.92, "k", u),
    ]
    subprons = [
        [
            {
                "pronunciation": ("t", "ʃ", "æ", "d"),
                "probability": None,
                "disambiguation": None,
                "right_sil_prob": None,
                "left_sil_prob": None,
                "left_nonsil_prob": None,
                "original_pronunciation": ("tʃ", "æ", "d"),
            }
        ],
        [
            {
                "pronunciation": ("l", "a", "ɪ", "k"),
                "probability": None,
                "disambiguation": None,
                "right_sil_prob": None,
                "left_sil_prob": None,
                "left_nonsil_prob": None,
                "original_pronunciation": ("l", "aɪ", "k"),
            }
        ],
    ]
    new_phones = d.data().map_to_original_pronunciation(cur_phones, subprons)
    assert new_phones == [
        CtmInterval(2.25, 2.43, "tʃ", u),
        CtmInterval(2.43, 2.55, "æ", u),
        CtmInterval(2.55, 2.64, "d", u),
        CtmInterval(2.64, 2.71, "l", u),
        CtmInterval(2.71, 2.84, "aɪ", u),
        CtmInterval(2.84, 2.92, "k", u),
    ]
