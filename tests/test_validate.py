import itertools
import os

from montreal_forced_aligner.validator import TrainingValidator


def test_training_validator_arpa(multilingual_ipa_tg_corpus_dir, english_dictionary, temp_dir):
    temp_dir = os.path.join(temp_dir, "training_validator")
    validator = TrainingValidator(
        corpus_directory=multilingual_ipa_tg_corpus_dir,
        dictionary_path=english_dictionary,
        temporary_directory=temp_dir,
        phone_set_type="ARPA",
    )
    validator.setup()
    d = validator.default_dictionary
    assert validator.phone_set_type.name == "ARPA"
    assert d.extra_questions_mapping
    assert d.phone_set_type.name == "ARPA"
    for k, v in d.extra_questions_mapping.items():
        print(k)
        print(v)
        assert len(v) == len(set(v))
    assert all("0" in x for x in d.extra_questions_mapping["stress_0"])
    assert all("1" in x for x in d.extra_questions_mapping["stress_1"])
    assert all("2" in x for x in d.extra_questions_mapping["stress_2"])
    assert "voiceless_fricative_variation" in d.extra_questions_mapping
    voiceless_fricatives = [
        x + p for x, p in itertools.product(["F", "HH", "K", "TH"], d.positions)
    ]
    assert all(
        x in d.extra_questions_mapping["voiceless_fricative_variation"]
        for x in voiceless_fricatives
    )
    assert set(d.extra_questions_mapping["high_back_variation"]) == {
        x + p
        for x, p in itertools.product(
            [
                "UH0",
                "UH1",
                "UH2",
                "UW0",
                "UW1",
                "UW2",
            ],
            d.positions,
        )
    }
    assert set(d.extra_questions_mapping["central_variation"]) == {
        x + p
        for x, p in itertools.product(
            [
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
            ],
            d.positions,
        )
    }

    topos = d.kaldi_phones_for_topo
    print(topos)
    assert 1 in topos
    assert 2 in topos
    assert "AY1_S" in topos[5]
    assert "JH_I" in topos[4]
    assert "B_E" in topos[2]
    assert "NG_B" in topos[3]
    assert set(topos[1]) == {
        x + p for x, p in itertools.product(["AH0", "ER0", "UH0", "IH0"], d.positions)
    }


def test_training_validator_ipa(
    multilingual_ipa_tg_corpus_dir, english_us_ipa_dictionary, temp_dir
):
    temp_dir = os.path.join(temp_dir, "training_validator_ipa")
    validator = TrainingValidator(
        corpus_directory=multilingual_ipa_tg_corpus_dir,
        dictionary_path=english_us_ipa_dictionary,
        temporary_directory=temp_dir,
        phone_set_type="IPA",
    )
    validator.setup()
    d = validator.default_dictionary
    assert validator.phone_set_type.name == "IPA"
    assert d.extra_questions_mapping
    assert d.phone_set_type.name == "IPA"
    for k, v in d.extra_questions_mapping.items():
        print(k)
        print(v)
        assert len(v) == len(set(v))
    assert "voiceless_fricative_variation" in d.extra_questions_mapping
    voiceless_fricatives = [
        x + p
        for x, p in itertools.product(
            [
                "θ",
                "f",
                "h",
            ],
            d.positions,
        )
    ]
    assert all(
        x in d.extra_questions_mapping["voiceless_fricative_variation"]
        for x in voiceless_fricatives
    )
    assert set(d.extra_questions_mapping["high_back_variation"]) == {
        x + p
        for x, p in itertools.product(
            [
                "ʊ",
                "u",
                "uː",
            ],
            d.positions,
        )
    }
    assert set(d.extra_questions_mapping["central_variation"]) == {
        x + p
        for x, p in itertools.product(
            [
                "ə",
                "ɚ",
                "ʌ",
                "ʊ",
                "ɝ",
                "ɝː",
            ],
            d.positions,
        )
    }

    topos = d.kaldi_phones_for_topo
    print(topos)
    assert 1 in topos
    assert 2 in topos
    assert "aɪ_S" in topos[5]
    assert "dʒ_I" in topos[4]
    assert "b_E" in topos[2]
    assert "ŋ_B" in topos[3]
    assert set(topos[1]) == {x + p for x, p in itertools.product(["ə", "ɚ", "ɾ"], d.positions)}
