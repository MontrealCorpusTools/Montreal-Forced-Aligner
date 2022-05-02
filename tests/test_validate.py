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
    assert validator.phone_set_type.name == "ARPA"
    assert validator.extra_questions_mapping
    assert validator.phone_set_type.name == "ARPA"
    for k, v in validator.extra_questions_mapping.items():
        print(k)
        print(v)
        assert len(v) == len(set(v))
    assert all("0" in x for x in validator.extra_questions_mapping["stress_0"])
    assert all("1" in x for x in validator.extra_questions_mapping["stress_1"])
    assert all("2" in x for x in validator.extra_questions_mapping["stress_2"])
    assert "fricatives" in validator.extra_questions_mapping
    fricatives = [
        x + p
        for x, p in itertools.product(
            {
                "V",
                "DH",
                "HH",
                "F",
                "TH",
            },
            validator.positions,
        )
    ]
    assert all(x in validator.extra_questions_mapping["fricatives"] for x in fricatives)
    assert set(validator.extra_questions_mapping["close"]) == {
        x + p
        for x, p in itertools.product(
            {"IH", "UH", "IY", "UW"},
            validator.positions,
        )
    }
    assert set(validator.extra_questions_mapping["close_mid"]) == {
        x + p
        for x, p in itertools.product(
            {"EY", "OW", "AH"},
            validator.positions,
        )
    }


def test_training_validator_ipa(
    multilingual_ipa_tg_corpus_dir, english_us_mfa_dictionary, temp_dir
):
    temp_dir = os.path.join(temp_dir, "training_validator_ipa")
    validator = TrainingValidator(
        corpus_directory=multilingual_ipa_tg_corpus_dir,
        dictionary_path=english_us_mfa_dictionary,
        temporary_directory=temp_dir,
        phone_set_type="IPA",
    )
    validator.setup()
    assert validator.phone_set_type.name == "IPA"
    assert validator.extra_questions_mapping
    assert validator.phone_set_type.name == "IPA"
    for k, v in validator.extra_questions_mapping.items():
        print(k)
        print(v)
        assert len(v) == len(set(v))
    assert "dental" in validator.extra_questions_mapping
    dental = {
        x + p
        for x, p in itertools.product(
            {"f", "v", "θ", "ð"},
            validator.positions,
        )
    }
    assert all(x in validator.extra_questions_mapping["dental"] for x in dental)
