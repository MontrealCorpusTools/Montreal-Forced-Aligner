import os
import shutil

from montreal_forced_aligner.db import Pronunciation
from montreal_forced_aligner.dictionary.multispeaker import MultispeakerDictionary


def test_abstract(abstract_dict_path, generated_dir, global_config, db_setup):
    output_directory = os.path.join(generated_dir, "dictionary_tests", "abstract")
    global_config.temporary_directory = output_directory
    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = MultispeakerDictionary(
        dictionary_path=abstract_dict_path, position_dependent_phones=True
    )
    dictionary.dictionary_setup()

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


def test_tabbed(tabbed_dict_path, basic_dict_path, generated_dir, global_config, db_setup):
    output_directory = os.path.join(generated_dir, "dictionary_tests", "tabbed")
    global_config.temporary_directory = output_directory
    shutil.rmtree(output_directory, ignore_errors=True)
    tabbed_dictionary = MultispeakerDictionary(dictionary_path=tabbed_dict_path)
    tabbed_dictionary.dictionary_setup()
    basic_dictionary = MultispeakerDictionary(dictionary_path=basic_dict_path)
    basic_dictionary.dictionary_setup()
    assert tabbed_dictionary.word_mapping(
        tabbed_dictionary.dictionary_lookup["test_tabbed_dictionary"]
    ) == basic_dictionary.word_mapping(basic_dictionary.dictionary_lookup["test_basic"])


def test_extra_annotations(extra_annotations_path, generated_dir, global_config, db_setup):
    output_directory = os.path.join(generated_dir, "dictionary_tests", "extras")
    global_config.temporary_directory = output_directory
    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = MultispeakerDictionary(dictionary_path=extra_annotations_path)
    dictionary.dictionary_setup()
    dictionary.write_lexicon_information()
    from montreal_forced_aligner.db import Grapheme

    with dictionary.session() as session:
        g = session.query(Grapheme).filter_by(grapheme="{").first()
        assert g is not None


def test_abstract_noposition(abstract_dict_path, generated_dir, global_config, db_setup):
    output_directory = os.path.join(generated_dir, "dictionary_tests", "abstract_no_position")
    global_config.temporary_directory = output_directory
    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = MultispeakerDictionary(
        dictionary_path=abstract_dict_path,
        position_dependent_phones=False,
    )
    dictionary.dictionary_setup()
    dictionary.write_lexicon_information()
    assert set(dictionary.phones) == {"sil", "spn", "phonea", "phoneb", "phonec"}


def test_english_clitics(english_dictionary, generated_dir, global_config, db_setup):
    output_directory = os.path.join(generated_dir, "dictionary_tests", "english_clitics")
    global_config.temporary_directory = output_directory
    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = MultispeakerDictionary(
        dictionary_path=english_dictionary,
        position_dependent_phones=False,
        phone_set_type="AUTO",
    )
    dictionary.dictionary_setup()
    dictionary.write_lexicon_information()
    assert dictionary.phone_set_type.name == "ARPA"
    assert dictionary.extra_questions_mapping
    for v in dictionary.extra_questions_mapping.values():
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


def test_english_mfa(english_us_mfa_dictionary, generated_dir, global_config, db_setup):
    output_directory = os.path.join(generated_dir, "dictionary_tests", "english_mfa")
    global_config.temporary_directory = output_directory
    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = MultispeakerDictionary(
        dictionary_path=english_us_mfa_dictionary,
        position_dependent_phones=False,
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


def test_mandarin_pinyin(pinyin_dictionary, generated_dir, global_config, db_setup):
    output_directory = os.path.join(generated_dir, "dictionary_tests", "pinyin")
    global_config.temporary_directory = output_directory
    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = MultispeakerDictionary(
        dictionary_path=pinyin_dictionary,
        position_dependent_phones=False,
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


def test_multispeaker_config(
    multispeaker_dictionary_config_path, generated_dir, global_config, db_setup
):
    output_directory = os.path.join(generated_dir, "dictionary_tests", "multispeaker")
    global_config.temporary_directory = output_directory
    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = MultispeakerDictionary(
        dictionary_path=multispeaker_dictionary_config_path,
        position_dependent_phones=False,
        punctuation=list(".-']["),
    )
    dictionary.dictionary_setup()
    dictionary.write_lexicon_information()


def test_mixed_dictionary(mixed_dict_path, generated_dir, global_config, db_setup):
    output_directory = os.path.join(generated_dir, "dictionary_tests", "mixed")
    global_config.temporary_directory = output_directory
    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = MultispeakerDictionary(
        dictionary_path=mixed_dict_path,
        position_dependent_phones=False,
    )

    dictionary.dictionary_setup()
    dictionary.write_lexicon_information()
    with dictionary.session() as session:
        pron = (
            session.query(Pronunciation).filter(Pronunciation.pronunciation == "dh ih s").first()
        )
        assert pron is not None
        assert pron.probability == 1.0
        assert pron.silence_after_probability == 0.43
        assert pron.silence_before_correction == 1.23
        assert pron.non_silence_before_correction == 0.85

        pron = (
            session.query(Pronunciation).filter(Pronunciation.pronunciation == "ay m ih").first()
        )
        assert pron is not None
        assert pron.probability == 0.01
        assert pron.silence_after_probability is None
        assert pron.silence_before_correction is None
        assert pron.non_silence_before_correction is None

        pron = session.query(Pronunciation).filter(Pronunciation.pronunciation == "dh ah").first()
        assert pron is not None
        assert pron.probability == 1
        assert pron.silence_after_probability == 0.5
        assert pron.silence_before_correction == 1.0
        assert pron.non_silence_before_correction == 1.0


def test_vietnamese_tones(vietnamese_dict_path, generated_dir, global_config, db_setup):
    output_directory = os.path.join(generated_dir, "dictionary_tests", "vietnamese")
    global_config.temporary_directory = output_directory
    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = MultispeakerDictionary(
        dictionary_path=vietnamese_dict_path,
        position_dependent_phones=False,
        phone_set_type="IPA",
    )
    dictionary.dictionary_setup()
    assert dictionary.get_base_phone("o˨˩ˀ") == "o"
    assert "o" in dictionary.kaldi_grouped_phones
    assert "o˨˩ˀ" in dictionary.kaldi_grouped_phones["o"]
    assert "o˦˩" in dictionary.kaldi_grouped_phones["o"]
