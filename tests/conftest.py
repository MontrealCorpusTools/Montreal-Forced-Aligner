from __future__ import annotations

import os
import shutil

import pytest
import yaml


@pytest.fixture(scope="session")
def test_dir():
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "data")


@pytest.fixture(scope="session")
def wav_dir(test_dir):
    return os.path.join(test_dir, "wav")


@pytest.fixture(scope="session")
def mp3_test_path(wav_dir):
    return os.path.join(wav_dir, "dummy.mp3")


@pytest.fixture(scope="session")
def opus_test_path(wav_dir):
    return os.path.join(wav_dir, "13697_11991_000000.opus")


@pytest.fixture(scope="session")
def lab_dir(test_dir):
    return os.path.join(test_dir, "lab")


@pytest.fixture(scope="session")
def textgrid_dir(test_dir):
    return os.path.join(test_dir, "textgrid")


@pytest.fixture(scope="session")
def acoustic_model_dir(test_dir):
    return os.path.join(test_dir, "am")


@pytest.fixture(scope="session")
def language_model_dir(test_dir):
    return os.path.join(test_dir, "lm")


@pytest.fixture(scope="session")
def generated_dir(test_dir):
    generated = os.path.join(test_dir, "generated")
    shutil.rmtree(generated, ignore_errors=True)
    if not os.path.exists(generated):
        os.makedirs(generated)
    return generated


@pytest.fixture(scope="session")
def temp_dir(generated_dir):

    return os.path.join(generated_dir, "temp")


@pytest.fixture(scope="session")
def model_manager():
    from montreal_forced_aligner.models import ModelManager

    return ModelManager()


@pytest.fixture(scope="session")
def english_acoustic_model(model_manager):
    if not model_manager.has_local_model("acoustic", "english_us_arpa"):
        model_manager.download_model("acoustic", "english_us_arpa")
    return "english_us_arpa"


@pytest.fixture(scope="session")
def english_dictionary(model_manager):
    if not model_manager.has_local_model("dictionary", "english_us_arpa"):
        model_manager.download_model("dictionary", "english_us_arpa")
    return "english_us_arpa"


@pytest.fixture(scope="session")
def german_prosodylab_acoustic_model(model_manager):
    if not model_manager.has_local_model("acoustic", "german_prosodylab"):
        model_manager.download_model("acoustic", "german_prosodylab")
    return "german_prosodylab"


@pytest.fixture(scope="session")
def german_prosodylab_dictionary(model_manager):
    if not model_manager.has_local_model("dictionary", "german_prosodylab"):
        model_manager.download_model("dictionary", "german_prosodylab")
    return "german_prosodylab"


@pytest.fixture(scope="session")
def english_mfa_acoustic_model(model_manager):
    if not model_manager.has_local_model("acoustic", "english_mfa"):
        model_manager.download_model("acoustic", "english_mfa")
    return "english_mfa"


@pytest.fixture(scope="session")
def english_us_mfa_dictionary(model_manager):
    if not model_manager.has_local_model("dictionary", "english_us_mfa"):
        model_manager.download_model("dictionary", "english_us_mfa")
    return "english_us_mfa"


@pytest.fixture(scope="session")
def english_us_mfa_dictionary_subset(english_us_mfa_dictionary, generated_dir):
    from montreal_forced_aligner.models import DictionaryModel

    path = os.path.join(generated_dir, "subset_english_us.dict")
    if not os.path.exists(path):
        model = DictionaryModel(english_us_mfa_dictionary)
        with open(model.path, "r", encoding="utf8") as inf, open(
            path, "w", encoding="utf8"
        ) as outf:
            for i, line in enumerate(inf):
                outf.write(line)
                if i >= 100:
                    break
    return path


@pytest.fixture(scope="session")
def swedish_mfa_acoustic_model(model_manager):
    if not model_manager.has_local_model("acoustic", "swedish_mfa"):
        model_manager.download_model("acoustic", "swedish_mfa")
    return "swedish_mfa"


@pytest.fixture(scope="session")
def swedish_cv_acoustic_model(model_manager):
    if not model_manager.has_local_model("acoustic", "swedish_cv"):
        model_manager.download_model("acoustic", "swedish_cv")
    return "swedish_cv"


@pytest.fixture(scope="session")
def swedish_mfa_dictionary(model_manager):
    if not model_manager.has_local_model("dictionary", "swedish_mfa"):
        model_manager.download_model("dictionary", "swedish_mfa")
    return "swedish_mfa"


@pytest.fixture(scope="session")
def swedish_cv_dictionary(model_manager):
    if not model_manager.has_local_model("dictionary", "swedish_cv"):
        model_manager.download_model("dictionary", "swedish_cv")
    return "swedish_cv"


@pytest.fixture(scope="session")
def pinyin_dictionary(model_manager):
    if not model_manager.has_local_model("dictionary", "mandarin_pinyin"):
        model_manager.download_model("dictionary", "mandarin_pinyin")
    return "mandarin_pinyin"


@pytest.fixture(scope="session")
def english_uk_mfa_dictionary(model_manager):
    if not model_manager.has_local_model("dictionary", "english_uk_mfa"):
        model_manager.download_model("dictionary", "english_uk_mfa")
    return "english_uk_mfa"


@pytest.fixture(scope="session")
def english_ivector_model(model_manager):
    return None
    if not model_manager.has_local_model("ivector", "english_ivector"):
        model_manager.download_model("ivector", "english_ivector")
    return "english_ivector"


@pytest.fixture(scope="session")
def english_g2p_model(model_manager):
    if not model_manager.has_local_model("g2p", "english_us_arpa"):
        model_manager.download_model("g2p", "english_us_arpa")
    return "english_us_arpa"


@pytest.fixture(scope="session")
def english_us_mfa_g2p_model(model_manager):
    if not model_manager.has_local_model("g2p", "english_us_mfa"):
        model_manager.download_model("g2p", "english_us_mfa")
    return "english_us_mfa"


@pytest.fixture(scope="session")
def transcription_acoustic_model(acoustic_model_dir):
    return os.path.join(acoustic_model_dir, "mono_model.zip")


@pytest.fixture(scope="session")
def transcription_language_model(language_model_dir, generated_dir):
    return os.path.join(language_model_dir, "test_lm.zip")


@pytest.fixture(scope="session")
def transcription_language_model_arpa(language_model_dir, generated_dir):
    return os.path.join(language_model_dir, "test_lm.arpa")


@pytest.fixture(scope="session")
def corpus_root_dir(generated_dir):
    return os.path.join(generated_dir, "constructed_test_corpora")


@pytest.fixture(scope="session")
def output_model_dir(generated_dir):
    return os.path.join(generated_dir, "output_models")


@pytest.fixture(scope="session")
def mono_align_model_path(output_model_dir):
    return os.path.join(output_model_dir, "mono_model.zip")


@pytest.fixture(scope="session")
def basic_corpus_dir(corpus_root_dir, wav_dir, lab_dir):
    path = os.path.join(corpus_root_dir, "basic")
    os.makedirs(path, exist_ok=True)
    names = [("michael", ["acoustic_corpus"]), ("sickmichael", ["cold_corpus", "cold_corpus3"])]
    for s, files in names:
        s_dir = os.path.join(path, s)
        os.makedirs(s_dir, exist_ok=True)
        for name in files:
            space_name = name.replace("_", " ")
            shutil.copyfile(
                os.path.join(wav_dir, name + ".wav"), os.path.join(s_dir, name + ".wav")
            )
            shutil.copyfile(
                os.path.join(wav_dir, name + ".wav"), os.path.join(s_dir, space_name + ".wav")
            )
            shutil.copyfile(
                os.path.join(lab_dir, name + ".lab"), os.path.join(s_dir, name + ".lab")
            )
            shutil.copyfile(
                os.path.join(lab_dir, name + ".lab"), os.path.join(s_dir, space_name + ".lab")
            )
    return path


@pytest.fixture(scope="session")
def basic_reference_dir(corpus_root_dir, wav_dir, textgrid_dir):
    path = os.path.join(corpus_root_dir, "basic_reference")
    os.makedirs(path, exist_ok=True)
    names = [("michael", ["acoustic_corpus"]), ("sickmichael", ["cold_corpus", "cold_corpus3"])]
    for s, files in names:
        s_dir = os.path.join(path, s)
        os.makedirs(s_dir, exist_ok=True)
        for name in files:
            shutil.copyfile(
                os.path.join(textgrid_dir, name + ".TextGrid"),
                os.path.join(s_dir, name + ".TextGrid"),
            )
    return path


@pytest.fixture(scope="session")
def xsampa_corpus_dir(corpus_root_dir, wav_dir, lab_dir):
    path = os.path.join(corpus_root_dir, "xsampa")
    os.makedirs(path, exist_ok=True)

    s_dir = os.path.join(path, "michael")
    os.makedirs(s_dir, exist_ok=True)
    shutil.copyfile(
        os.path.join(wav_dir, "acoustic_corpus.wav"), os.path.join(s_dir, "xsampa.wav")
    )
    shutil.copyfile(os.path.join(lab_dir, "xsampa.lab"), os.path.join(s_dir, "xsampa.lab"))
    return path


@pytest.fixture(scope="session")
def basic_split_dir(corpus_root_dir, wav_dir, lab_dir, textgrid_dir):
    path = os.path.join(corpus_root_dir, "split")
    audio_path = os.path.join(path, "audio")
    text_path = os.path.join(path, "text")
    os.makedirs(path, exist_ok=True)
    names = [
        ("michael", ["acoustic_corpus"]),
        ("sickmichael", ["cold_corpus", "cold_corpus3"]),
        (
            "speaker",
            [
                "multilingual_ipa",
                "multilingual_ipa_2",
                "multilingual_ipa_3",
                "multilingual_ipa_4",
                "multilingual_ipa_5",
            ],
        ),
        (
            "speaker_two",
            [
                "multilingual_ipa_us",
                "multilingual_ipa_us_2",
                "multilingual_ipa_us_3",
                "multilingual_ipa_us_4",
                "multilingual_ipa_us_5",
            ],
        ),
    ]
    for s, files in names:
        s_text_dir = os.path.join(text_path, s)
        s_audio_dir = os.path.join(audio_path, s)
        os.makedirs(s_text_dir, exist_ok=True)
        os.makedirs(s_audio_dir, exist_ok=True)
        for name in files:
            wav_path = os.path.join(wav_dir, name + ".wav")
            if os.path.exists(wav_path):
                shutil.copyfile(wav_path, wav_path.replace(wav_dir, s_audio_dir))
            lab_path = os.path.join(lab_dir, name + ".lab")
            if not os.path.exists(lab_path):
                lab_path = os.path.join(lab_dir, name + ".txt")
            shutil.copyfile(lab_path, lab_path.replace(lab_dir, s_text_dir))
    return audio_path, text_path


@pytest.fixture(scope="session")
def multilingual_ipa_corpus_dir(corpus_root_dir, wav_dir, lab_dir):
    path = os.path.join(corpus_root_dir, "multilingual")
    os.makedirs(path, exist_ok=True)
    names = [
        (
            "speaker",
            [
                "multilingual_ipa",
                "multilingual_ipa_2",
                "multilingual_ipa_3",
                "multilingual_ipa_4",
                "multilingual_ipa_5",
            ],
        ),
        (
            "speaker_two",
            [
                "multilingual_ipa_us",
                "multilingual_ipa_us_2",
                "multilingual_ipa_us_3",
                "multilingual_ipa_us_4",
                "multilingual_ipa_us_5",
            ],
        ),
    ]
    for s, files in names:
        s_dir = os.path.join(path, s)
        os.makedirs(s_dir, exist_ok=True)
        for name in files:
            shutil.copyfile(
                os.path.join(wav_dir, name + ".flac"), os.path.join(s_dir, name + ".flac")
            )
            shutil.copyfile(
                os.path.join(lab_dir, name + ".txt"), os.path.join(s_dir, name + ".txt")
            )
    return path


@pytest.fixture(scope="session")
def multilingual_ipa_tg_corpus_dir(corpus_root_dir, wav_dir, textgrid_dir):
    path = os.path.join(corpus_root_dir, "multilingual_tg")
    os.makedirs(path, exist_ok=True)
    names = [
        (
            "speaker_one",
            [
                "multilingual_ipa",
                "multilingual_ipa_2",
                "multilingual_ipa_3",
                "multilingual_ipa_4",
                "multilingual_ipa_5",
            ],
        ),
        (
            "speaker_two",
            [
                "multilingual_ipa_us",
                "multilingual_ipa_us_2",
                "multilingual_ipa_us_3",
                "multilingual_ipa_us_4",
                "multilingual_ipa_us_5",
            ],
        ),
    ]
    for s, files in names:
        s_dir = os.path.join(path, s)
        os.makedirs(s_dir, exist_ok=True)
        for name in files:
            shutil.copyfile(
                os.path.join(wav_dir, name + ".flac"), os.path.join(s_dir, name + ".flac")
            )
            shutil.copyfile(
                os.path.join(textgrid_dir, name + ".TextGrid"),
                os.path.join(s_dir, name + ".TextGrid"),
            )
    return path


@pytest.fixture(scope="session")
def weird_words_dir(corpus_root_dir, wav_dir, lab_dir):
    path = os.path.join(corpus_root_dir, "weird_words")
    os.makedirs(path, exist_ok=True)
    name = "weird_words"
    shutil.copyfile(
        os.path.join(wav_dir, "acoustic_corpus.wav"), os.path.join(path, name + ".wav")
    )
    shutil.copyfile(os.path.join(lab_dir, name + ".lab"), os.path.join(path, name + ".lab"))
    return path


@pytest.fixture(scope="session")
def punctuated_dir(corpus_root_dir, wav_dir, lab_dir):
    path = os.path.join(corpus_root_dir, "punctuated")
    os.makedirs(path, exist_ok=True)
    name = "punctuated"
    shutil.copyfile(
        os.path.join(wav_dir, "acoustic_corpus.wav"), os.path.join(path, name + ".wav")
    )
    shutil.copyfile(os.path.join(lab_dir, name + ".lab"), os.path.join(path, name + ".lab"))
    name = "weird_words"
    shutil.copyfile(
        os.path.join(wav_dir, "acoustic_corpus.wav"), os.path.join(path, name + ".wav")
    )
    shutil.copyfile(os.path.join(lab_dir, name + ".lab"), os.path.join(path, name + ".lab"))
    return path


@pytest.fixture(scope="session")
def swedish_dir(corpus_root_dir, wav_dir, lab_dir):
    path = os.path.join(corpus_root_dir, "swedish")
    os.makedirs(path, exist_ok=True)
    names = [
        (
            "se10x016",
            [
                "se10x016-08071999-1334_u0016001",
                "se10x016-08071999-1334_u0016002",
                "se10x016-08071999-1334_u0016003",
                "se10x016-08071999-1334_u0016004",
            ],
        )
    ]
    for s, files in names:
        s_dir = os.path.join(path, s)
        os.makedirs(s_dir, exist_ok=True)
        for name in files:
            shutil.copyfile(
                os.path.join(wav_dir, name + ".wav"), os.path.join(s_dir, name + ".wav")
            )
            shutil.copyfile(
                os.path.join(lab_dir, name + ".lab"), os.path.join(s_dir, name + ".txt")
            )
    return path


@pytest.fixture(scope="session")
def basic_corpus_txt_dir(corpus_root_dir, wav_dir, lab_dir):
    path = os.path.join(corpus_root_dir, "basic_txt")
    os.makedirs(path, exist_ok=True)
    names = [("michael", ["acoustic_corpus"]), ("sickmichael", ["cold_corpus", "cold_corpus3"])]
    for s, files in names:
        s_dir = os.path.join(path, s)
        os.makedirs(s_dir, exist_ok=True)
        for name in files:
            shutil.copyfile(
                os.path.join(wav_dir, name + ".wav"), os.path.join(s_dir, name + ".wav")
            )
            shutil.copyfile(
                os.path.join(lab_dir, name + ".lab"), os.path.join(s_dir, name + ".txt")
            )
    return path


@pytest.fixture(scope="session")
def extra_corpus_dir(corpus_root_dir, wav_dir, lab_dir):
    path = os.path.join(corpus_root_dir, "extra")
    os.makedirs(path, exist_ok=True)
    name = "cold_corpus3"
    shutil.copyfile(os.path.join(wav_dir, name + ".wav"), os.path.join(path, name + ".wav"))
    shutil.copyfile(os.path.join(lab_dir, name + "_extra.lab"), os.path.join(path, name + ".lab"))
    return path


@pytest.fixture(scope="session")
def transcribe_corpus_24bit_dir(corpus_root_dir, wav_dir):
    path = os.path.join(corpus_root_dir, "24bit")
    os.makedirs(path, exist_ok=True)
    name = "cold_corpus_24bit"
    shutil.copyfile(os.path.join(wav_dir, name + ".wav"), os.path.join(path, name + ".wav"))
    name = "cold_corpus_32bit_float"
    shutil.copyfile(os.path.join(wav_dir, name + ".wav"), os.path.join(path, name + ".wav"))
    return path


@pytest.fixture(scope="session")
def stereo_corpus_dir(corpus_root_dir, wav_dir, textgrid_dir):
    path = os.path.join(corpus_root_dir, "stereo")
    os.makedirs(path, exist_ok=True)
    name = "michaelandsickmichael"
    shutil.copyfile(os.path.join(wav_dir, name + ".wav"), os.path.join(path, name + ".wav"))
    shutil.copyfile(
        os.path.join(textgrid_dir, name + ".TextGrid"), os.path.join(path, name + ".TextGrid")
    )
    return path


@pytest.fixture(scope="session")
def mp3_corpus_dir(corpus_root_dir, wav_dir, lab_dir):
    path = os.path.join(corpus_root_dir, "cv_mp3")
    os.makedirs(path, exist_ok=True)
    names = ["common_voice_en_22058264", "common_voice_en_22058266", "common_voice_en_22058267"]
    for name in names:
        shutil.copyfile(os.path.join(wav_dir, name + ".mp3"), os.path.join(path, name + ".mp3"))
        shutil.copyfile(os.path.join(lab_dir, name + ".lab"), os.path.join(path, name + ".lab"))
    return path


@pytest.fixture(scope="session")
def opus_corpus_dir(corpus_root_dir, wav_dir, lab_dir):
    path = os.path.join(corpus_root_dir, "mls_opus")
    os.makedirs(path, exist_ok=True)
    names = ["13697_11991_000000"]
    for name in names:
        shutil.copyfile(os.path.join(wav_dir, name + ".opus"), os.path.join(path, name + ".opus"))
        shutil.copyfile(os.path.join(lab_dir, name + ".lab"), os.path.join(path, name + ".lab"))
    return path


@pytest.fixture(scope="session")
def stereo_corpus_short_tg_dir(corpus_root_dir, wav_dir, textgrid_dir):
    path = os.path.join(corpus_root_dir, "stereo_short_tg")
    os.makedirs(path, exist_ok=True)
    name = "michaelandsickmichael"
    shutil.copyfile(os.path.join(wav_dir, name + ".wav"), os.path.join(path, name + ".wav"))
    shutil.copyfile(
        os.path.join(textgrid_dir, name + "_short_tg.TextGrid"),
        os.path.join(path, name + ".TextGrid"),
    )
    return path


@pytest.fixture(scope="session")
def flac_corpus_dir(corpus_root_dir, wav_dir, lab_dir):
    path = os.path.join(corpus_root_dir, "flac_corpus")
    os.makedirs(path, exist_ok=True)
    name = "61-70968-0000"
    shutil.copyfile(os.path.join(wav_dir, name + ".flac"), os.path.join(path, name + ".flac"))
    shutil.copyfile(os.path.join(lab_dir, name + ".lab"), os.path.join(path, name + ".lab"))
    return path


@pytest.fixture(scope="session")
def flac_tg_corpus_dir(corpus_root_dir, wav_dir, textgrid_dir):
    path = os.path.join(corpus_root_dir, "flac_tg_corpus")
    os.makedirs(path, exist_ok=True)
    name = "61-70968-0000"
    shutil.copyfile(os.path.join(wav_dir, name + ".flac"), os.path.join(path, name + ".flac"))
    shutil.copyfile(
        os.path.join(textgrid_dir, name + ".TextGrid"), os.path.join(path, name + ".TextGrid")
    )
    return path


@pytest.fixture(scope="session")
def shortsegments_corpus_dir(corpus_root_dir, wav_dir, textgrid_dir):
    path = os.path.join(corpus_root_dir, "short_segments")
    os.makedirs(path, exist_ok=True)
    name = "short_segments"
    shutil.copyfile(os.path.join(wav_dir, "dummy.wav"), os.path.join(path, name + ".wav"))
    shutil.copyfile(
        os.path.join(textgrid_dir, name + ".TextGrid"), os.path.join(path, name + ".TextGrid")
    )
    return path


@pytest.fixture(scope="session")
def dict_dir(test_dir):
    return os.path.join(test_dir, "dictionaries")


@pytest.fixture(scope="session")
def abstract_dict_path(dict_dir):
    return os.path.join(dict_dir, "abstract.txt")


@pytest.fixture(scope="session")
def basic_dict_path(dict_dir):
    return os.path.join(dict_dir, "basic.txt")


@pytest.fixture(scope="session")
def tabbed_dict_path(dict_dir):
    return os.path.join(dict_dir, "tabbed_dictionary.txt")


@pytest.fixture(scope="session")
def extra_annotations_path(dict_dir):
    return os.path.join(dict_dir, "extra_annotations.txt")


@pytest.fixture(scope="session")
def frclitics_dict_path(dict_dir):
    return os.path.join(dict_dir, "frclitics.txt")


@pytest.fixture(scope="session")
def xsampa_dict_path(dict_dir):
    return os.path.join(dict_dir, "xsampa.txt")


@pytest.fixture(scope="session")
def vietnamese_dict_path(dict_dir):
    return os.path.join(dict_dir, "vietnamese_ipa.txt")


@pytest.fixture(scope="session")
def acoustic_dict_path(dict_dir):
    return os.path.join(dict_dir, "acoustic.txt")


@pytest.fixture(scope="session")
def speaker_dictionary_path(basic_dict_path, acoustic_dict_path, generated_dir):
    data = {"default": acoustic_dict_path, "sickmichael": basic_dict_path}
    speaker_dict_path = os.path.join(generated_dir, "basic_acoustic_dicts.yaml")
    with open(speaker_dict_path, "w") as f:
        yaml.safe_dump(data, f)
    return speaker_dict_path


@pytest.fixture(scope="session")
def mono_output_directory(generated_dir):
    return os.path.join(generated_dir, "mono_output")


@pytest.fixture(scope="session")
def textgrid_output_model_path(generated_dir):
    return os.path.join(generated_dir, "textgrid_output_model.zip")


@pytest.fixture(scope="session")
def acoustic_g2p_model_path(generated_dir):
    return os.path.join(generated_dir, "acoustic_g2p_output_model.zip")


@pytest.fixture(scope="session")
def ivector_output_model_path(generated_dir):
    return os.path.join(generated_dir, "ivector_output_model.zip")


@pytest.fixture(scope="session")
def basic_g2p_model_path(generated_dir):
    return os.path.join(generated_dir, "basic_g2p.zip")


@pytest.fixture(scope="session")
def g2p_basic_output(generated_dir):
    return os.path.join(generated_dir, "g2p_basic.txt")


@pytest.fixture(scope="session")
def orth_basic_output(generated_dir):
    return os.path.join(generated_dir, "orth_basic.txt")


@pytest.fixture(scope="session")
def config_directory(test_dir):
    return os.path.join(test_dir, "configs")


@pytest.fixture(scope="session")
def eval_mapping_path(config_directory):
    return os.path.join(config_directory, "eval_mapping.yaml")


@pytest.fixture(scope="session")
def basic_train_config_path(config_directory):
    return os.path.join(config_directory, "basic_train_config.yaml")


@pytest.fixture(scope="session")
def transcribe_config_path(config_directory):
    return os.path.join(config_directory, "transcribe.yaml")


@pytest.fixture(scope="session")
def g2p_config_path(config_directory):
    return os.path.join(config_directory, "g2p_config.yaml")


@pytest.fixture(scope="session")
def train_g2p_config_path(config_directory):
    return os.path.join(config_directory, "train_g2p_config.yaml")


@pytest.fixture(scope="session")
def basic_train_lm_config_path(config_directory):
    return os.path.join(config_directory, "basic_train_lm.yaml")


@pytest.fixture(scope="session")
def different_punctuation_config_path(config_directory):
    return os.path.join(config_directory, "different_punctuation_config.yaml")


@pytest.fixture(scope="session")
def no_punctuation_config_path(config_directory):
    return os.path.join(config_directory, "no_punctuation_config.yaml")


@pytest.fixture(scope="session")
def basic_align_config_path(config_directory):
    return os.path.join(config_directory, "basic_align_config.yaml")


@pytest.fixture(scope="session")
def basic_segment_config_path(config_directory):
    return os.path.join(config_directory, "basic_segment_config.yaml")


@pytest.fixture(scope="session")
def train_ivector_config_path(config_directory):
    return os.path.join(config_directory, "ivector_train.yaml")


@pytest.fixture(scope="session")
def mono_align_config_path(config_directory):
    return os.path.join(config_directory, "mono_align.yaml")


@pytest.fixture(scope="session")
def pron_train_config_path(config_directory):
    return os.path.join(config_directory, "pron_train.yaml")


@pytest.fixture(scope="session")
def mono_train_config_path(config_directory):
    return os.path.join(config_directory, "mono_train.yaml")


@pytest.fixture(scope="session")
def tri_train_config_path(config_directory):
    return os.path.join(config_directory, "tri_train.yaml")


@pytest.fixture(scope="session")
def pitch_train_config_path(config_directory):
    return os.path.join(config_directory, "pitch_tri_train.yaml")


@pytest.fixture(scope="session")
def lda_train_config_path(config_directory):
    return os.path.join(config_directory, "lda_train.yaml")


@pytest.fixture(scope="session")
def sat_train_config_path(config_directory):
    return os.path.join(config_directory, "sat_train.yaml")


@pytest.fixture(scope="session")
def multispeaker_dictionary_config_path(generated_dir, basic_dict_path, english_dictionary):
    path = os.path.join(generated_dir, "multispeaker_dictionary.yaml")
    with open(path, "w", encoding="utf8") as f:
        yaml.safe_dump({"default": english_dictionary, "michael": basic_dict_path}, f)
    return path


@pytest.fixture(scope="session")
def mfa_speaker_dict_path(generated_dir, english_uk_mfa_dictionary, english_us_mfa_dictionary):
    path = os.path.join(generated_dir, "multispeaker_mfa_dictionary.yaml")
    with open(path, "w", encoding="utf8") as f:
        yaml.safe_dump(
            {"default": english_us_mfa_dictionary, "speaker": english_uk_mfa_dictionary}, f
        )
    return path


@pytest.fixture(scope="session")
def test_align_config():
    return {"beam": 100, "retry_beam": 400}
