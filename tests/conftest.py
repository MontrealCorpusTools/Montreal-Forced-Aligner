from __future__ import annotations

import os
import pathlib
import shutil

import mock
import pytest
import sqlalchemy.orm
import yaml

from montreal_forced_aligner import config
from montreal_forced_aligner.config import get_temporary_directory
from montreal_forced_aligner.helper import mfa_open


@pytest.fixture(autouse=True, scope="session")
def mock_settings_env_vars():
    with mock.patch.dict(os.environ, {"MFA_PROFILE": "test", "SQLALCHEMY_WARN_20": "true"}):
        yield


@pytest.fixture(scope="session")
def test_dir():
    base = pathlib.Path(__file__).parent
    return base.joinpath("data")


@pytest.fixture(scope="session")
def wav_dir(test_dir):
    return test_dir.joinpath("wav")


@pytest.fixture(scope="session")
def mp3_test_path(wav_dir):
    return wav_dir.joinpath("dummy.mp3")


@pytest.fixture(scope="session")
def opus_test_path(wav_dir):
    return wav_dir.joinpath("13697_11991_000000.opus")


@pytest.fixture(scope="session")
def lab_dir(test_dir):
    return test_dir.joinpath("lab")


@pytest.fixture(scope="session")
def textgrid_dir(test_dir):
    return test_dir.joinpath("textgrid")


@pytest.fixture(scope="session")
def acoustic_model_dir(test_dir):
    return test_dir.joinpath("am")


@pytest.fixture(scope="session")
def tokenizer_model_dir(test_dir):
    return test_dir.joinpath("tokenizer")


@pytest.fixture(scope="session")
def language_model_dir(test_dir):
    return test_dir.joinpath("lm")


@pytest.fixture(scope="session")
def generated_dir(test_dir):
    generated = test_dir.joinpath("generated")
    shutil.rmtree(generated, ignore_errors=True)
    generated.mkdir(parents=True, exist_ok=True)
    return generated


@pytest.fixture(scope="session")
def global_config():
    config.CURRENT_PROFILE_NAME = "test"
    config.CLEAN = True
    config.USE_POSTGRES = True
    config.DEBUG = True
    config.VERBOSE = False
    config.NUM_JOBS = 2
    config.USE_MP = True
    config.DATABASE_LIMITED_MODE = True
    config.AUTO_SERVER = False
    config.TEMPORARY_DIRECTORY = get_temporary_directory()


@pytest.fixture(scope="session")
def temp_dir(global_config):
    yield config.TEMPORARY_DIRECTORY


@pytest.fixture(scope="function")
def db_setup(temp_dir, request):
    sqlalchemy.orm.close_all_sessions()
    return True


@pytest.fixture(scope="session")
def model_manager():
    from montreal_forced_aligner.models import ModelManager

    github_token = os.getenv("GITHUB_TOKEN", None)
    return ModelManager(github_token)


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

    path = generated_dir.joinpath("subset_english_us.dict")
    if not os.path.exists(path):
        model = DictionaryModel(english_us_mfa_dictionary)
        with mfa_open(model.path, "r") as inf, mfa_open(path, "w") as outf:
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
    if not model_manager.has_local_model("ivector", "english_mfa"):
        model_manager.download_model("ivector", "english_mfa")
    return "english_mfa"


@pytest.fixture(scope="session")
def multilingual_ivector_model(model_manager):
    if not model_manager.has_local_model("ivector", "multilingual_mfa"):
        model_manager.download_model("ivector", "multilingual_mfa")
    return "multilingual_mfa"


@pytest.fixture(scope="session")
def english_g2p_model(model_manager):
    if not model_manager.has_local_model("g2p", "english_us_arpa"):
        model_manager.download_model("g2p", "english_us_arpa")
    return "english_us_arpa"


@pytest.fixture(scope="session")
def japanese_tokenizer_model(model_manager):
    if not model_manager.has_local_model("tokenizer", "japanese_mfa"):
        model_manager.download_model("tokenizer", "japanese_mfa")
    return "japanese_mfa"


@pytest.fixture(scope="session")
def english_us_mfa_g2p_model(model_manager):
    if not model_manager.has_local_model("g2p", "english_us_mfa"):
        model_manager.download_model("g2p", "english_us_mfa")
    return "english_us_mfa"


@pytest.fixture(scope="session")
def transcription_acoustic_model(acoustic_model_dir):
    return acoustic_model_dir.joinpath("mono_model.zip")


@pytest.fixture(scope="session")
def test_tokenizer_model(tokenizer_model_dir):
    return tokenizer_model_dir.joinpath("test_tokenizer_model.zip")


@pytest.fixture(scope="session")
def test_tokenizer_model_phonetisaurus(tokenizer_model_dir):
    return tokenizer_model_dir.joinpath("test_tokenizer_model_phonetisaurus.zip")


@pytest.fixture(scope="session")
def transcription_language_model(language_model_dir, generated_dir):
    return language_model_dir.joinpath("test_lm.zip")


@pytest.fixture(scope="session")
def transcription_language_model_arpa(language_model_dir, generated_dir):
    return language_model_dir.joinpath("test_lm.arpa")


@pytest.fixture(scope="session")
def corpus_root_dir(generated_dir):
    return generated_dir.joinpath("constructed_test_corpora")


@pytest.fixture(scope="session")
def output_model_dir(generated_dir):
    return generated_dir.joinpath("output_models")


@pytest.fixture(scope="session")
def mono_align_model_path(output_model_dir):
    return output_model_dir.joinpath("mono_model.zip")


@pytest.fixture()
def basic_corpus_dir(corpus_root_dir, wav_dir, lab_dir):
    path = corpus_root_dir.joinpath("test_basic")
    path.mkdir(parents=True, exist_ok=True)
    names = [("michael", ["acoustic_corpus"]), ("sickmichael", ["cold_corpus", "cold_corpus3"])]
    for s, files in names:
        s_dir = path.joinpath(s)
        s_dir.mkdir(exist_ok=True)
        for name in files:
            space_name = name.replace("_", " ")
            shutil.copyfile(wav_dir.joinpath(name + ".wav"), s_dir.joinpath(name + ".wav"))
            shutil.copyfile(wav_dir.joinpath(name + ".wav"), s_dir.joinpath(space_name + ".wav"))
            shutil.copyfile(lab_dir.joinpath(name + ".lab"), s_dir.joinpath(name + ".lab"))
            shutil.copyfile(lab_dir.joinpath(name + ".lab"), s_dir.joinpath(space_name + ".lab"))
    return path


@pytest.fixture()
def combined_corpus_dir(corpus_root_dir, wav_dir, lab_dir):
    path = corpus_root_dir.joinpath("test_combined")
    path.mkdir(parents=True, exist_ok=True)
    os.makedirs(path, exist_ok=True)
    names = [
        ("michael", ["acoustic_corpus.wav"]),
        ("sickmichael", ["cold_corpus.wav", "cold_corpus3.wav"]),
        (
            "speaker",
            [
                "multilingual_ipa.flac",
                "multilingual_ipa_2.flac",
                "multilingual_ipa_3.flac",
                "multilingual_ipa_4.flac",
                "multilingual_ipa_5.flac",
            ],
        ),
        (
            "speaker_two",
            [
                "multilingual_ipa_us.flac",
                "multilingual_ipa_us_2.flac",
                "multilingual_ipa_us_3.flac",
                "multilingual_ipa_us_4.flac",
                "multilingual_ipa_us_5.flac",
            ],
        ),
        (
            "speaker_three",
            [
                "common_voice_en_22058264.mp3",
                "common_voice_en_22058266.mp3",
                "common_voice_en_22058267.mp3",
            ],
        ),
    ]
    for s, files in names:
        s_dir = path.joinpath(s)
        os.makedirs(s_dir, exist_ok=True)
        for name in files:
            shutil.copyfile(wav_dir.joinpath(name), s_dir.joinpath(name))
            text_name = name.split(".")[0] + ".lab"
            if not lab_dir.joinpath(text_name).exists():
                text_name = name.split(".")[0] + ".txt"
            shutil.copyfile(lab_dir.joinpath(text_name), s_dir.joinpath(text_name))
    return path


@pytest.fixture()
def duplicated_name_corpus_dir(corpus_root_dir, wav_dir, lab_dir):
    path = corpus_root_dir.joinpath("test_duplicated")
    os.makedirs(path, exist_ok=True)
    names = [("michael", ["acoustic_corpus"]), ("sickmichael", ["cold_corpus", "cold_corpus3"])]
    for s, files in names:
        s_dir = path.joinpath(s)
        os.makedirs(s_dir, exist_ok=True)
        for i, name in enumerate(files):
            new_name = f"recording_{i}"
            shutil.copyfile(wav_dir.joinpath(name + ".wav"), s_dir.joinpath(new_name + ".wav"))
            shutil.copyfile(lab_dir.joinpath(name + ".lab"), s_dir.joinpath(new_name + ".lab"))
    return path


@pytest.fixture(scope="session")
def basic_reference_dir(corpus_root_dir, wav_dir, textgrid_dir):
    path = corpus_root_dir.joinpath("test_basic_reference")
    os.makedirs(path, exist_ok=True)
    names = [("michael", ["acoustic_corpus"]), ("sickmichael", ["cold_corpus", "cold_corpus3"])]
    for s, files in names:
        s_dir = path.joinpath(s)
        os.makedirs(s_dir, exist_ok=True)
        for name in files:
            shutil.copyfile(
                textgrid_dir.joinpath(name + ".TextGrid"),
                s_dir.joinpath(name + ".TextGrid"),
            )
    return path


@pytest.fixture()
def xsampa_corpus_dir(corpus_root_dir, wav_dir, lab_dir):
    path = corpus_root_dir.joinpath("test_xsampa")
    os.makedirs(path, exist_ok=True)

    s_dir = path.joinpath("michael")
    os.makedirs(s_dir, exist_ok=True)
    shutil.copyfile(wav_dir.joinpath("acoustic_corpus.wav"), s_dir.joinpath("xsampa.wav"))
    shutil.copyfile(lab_dir.joinpath("xsampa.lab"), s_dir.joinpath("xsampa.lab"))
    return path


@pytest.fixture()
def basic_split_dir(corpus_root_dir, wav_dir, lab_dir, textgrid_dir):
    path = corpus_root_dir.joinpath("test_split")
    audio_path = path.joinpath("audio")
    text_path = path.joinpath("text")
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
        s_text_dir = text_path.joinpath(s)
        s_audio_dir = audio_path.joinpath(s)
        os.makedirs(s_text_dir, exist_ok=True)
        os.makedirs(s_audio_dir, exist_ok=True)
        for name in files:
            wav_path = wav_dir.joinpath(name + ".wav")
            if os.path.exists(wav_path):
                shutil.copyfile(wav_path, s_audio_dir.joinpath(name + ".wav"))
            lab_path = lab_dir.joinpath(name + ".lab")
            if not os.path.exists(lab_path):
                lab_path = lab_dir.joinpath(name + ".txt")
            shutil.copyfile(lab_path, s_text_dir.joinpath(lab_path.name))
    return audio_path, text_path


@pytest.fixture()
def multilingual_ipa_corpus_dir(corpus_root_dir, wav_dir, lab_dir):
    path = corpus_root_dir.joinpath("test_multilingual")
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
        s_dir = path.joinpath(s)
        os.makedirs(s_dir, exist_ok=True)
        for name in files:
            shutil.copyfile(wav_dir.joinpath(name + ".flac"), s_dir.joinpath(name + ".flac"))
            shutil.copyfile(lab_dir.joinpath(name + ".txt"), s_dir.joinpath(name + ".txt"))
    return path


@pytest.fixture()
def multilingual_ipa_tg_corpus_dir(corpus_root_dir, wav_dir, textgrid_dir):
    path = corpus_root_dir.joinpath("test_multilingual_tg")
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
        s_dir = path.joinpath(s)
        os.makedirs(s_dir, exist_ok=True)
        for name in files:
            shutil.copyfile(wav_dir.joinpath(name + ".flac"), s_dir.joinpath(name + ".flac"))
            shutil.copyfile(
                textgrid_dir.joinpath(name + ".TextGrid"),
                s_dir.joinpath(name + ".TextGrid"),
            )
    return path


@pytest.fixture()
def weird_words_dir(corpus_root_dir, wav_dir, lab_dir):
    path = corpus_root_dir.joinpath("test_weird_words")
    os.makedirs(path, exist_ok=True)
    name = "weird_words"
    shutil.copyfile(wav_dir.joinpath("acoustic_corpus.wav"), path.joinpath(name + ".wav"))
    shutil.copyfile(lab_dir.joinpath(name + ".lab"), path.joinpath(name + ".lab"))
    return path


@pytest.fixture()
def punctuated_dir(corpus_root_dir, wav_dir, lab_dir):
    path = corpus_root_dir.joinpath("test_punctuated")
    os.makedirs(path, exist_ok=True)
    name = "punctuated"
    shutil.copyfile(wav_dir.joinpath("acoustic_corpus.wav"), path.joinpath(name + ".wav"))
    shutil.copyfile(lab_dir.joinpath(name + ".lab"), path.joinpath(name + ".lab"))
    name = "weird_words"
    shutil.copyfile(wav_dir.joinpath("acoustic_corpus.wav"), path.joinpath(name + ".wav"))
    shutil.copyfile(lab_dir.joinpath(name + ".lab"), path.joinpath(name + ".lab"))
    return path


@pytest.fixture()
def japanese_dir(corpus_root_dir, wav_dir, lab_dir):
    path = corpus_root_dir.joinpath("test_japanese")
    os.makedirs(path, exist_ok=True)
    name = "日本語"
    shutil.copyfile(lab_dir.joinpath(name + ".lab"), path.joinpath(name + ".lab"))
    return path


@pytest.fixture()
def devanagari_dir(corpus_root_dir, wav_dir, lab_dir):
    path = corpus_root_dir.joinpath("test_devanagari")
    os.makedirs(path, exist_ok=True)
    name = "devanagari"
    shutil.copyfile(lab_dir.joinpath(name + ".lab"), path.joinpath(name + ".lab"))
    return path


@pytest.fixture()
def french_clitics_dir(corpus_root_dir, wav_dir, lab_dir):
    path = corpus_root_dir.joinpath("test_french_clitics")
    os.makedirs(path, exist_ok=True)
    name = "french_clitics"
    shutil.copyfile(lab_dir.joinpath(name + ".lab"), path.joinpath(name + ".lab"))
    return path


@pytest.fixture()
def swedish_dir(corpus_root_dir, wav_dir, lab_dir):
    path = corpus_root_dir.joinpath("test_swedish")
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
        s_dir = path.joinpath(s)
        os.makedirs(s_dir, exist_ok=True)
        for name in files:
            shutil.copyfile(wav_dir.joinpath(name + ".wav"), s_dir.joinpath(name + ".wav"))
            shutil.copyfile(lab_dir.joinpath(name + ".lab"), s_dir.joinpath(name + ".txt"))
    return path


@pytest.fixture()
def japanese_cv_dir(corpus_root_dir, wav_dir, lab_dir):
    path = corpus_root_dir.joinpath("test_japanese_cv")
    path.mkdir(parents=True, exist_ok=True)
    names = [
        (
            "02a8841a00d762472a4797b56ee01643e8d9ece5a225f2e91c007ab1f94c49c99e50d19986ff3fefb18190257323f34238828114aa607f84fbe9764ecf5aaeaa",
            [
                "common_voice_ja_24511055",
            ],
        )
    ]
    for s, files in names:
        s_dir = path.joinpath(s)
        s_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            shutil.copyfile(wav_dir.joinpath(name + ".mp3"), s_dir.joinpath(name + ".mp3"))
            shutil.copyfile(lab_dir.joinpath(name + ".lab"), s_dir.joinpath(name + ".lab"))
    return path


@pytest.fixture()
def japanese_cv_japanese_name_dir(corpus_root_dir, wav_dir, lab_dir):
    path = corpus_root_dir.joinpath("test_japanese_cv")
    path.mkdir(parents=True, exist_ok=True)
    names = [
        (
            "だれか",
            [
                "common_voice_ja_24511055",
            ],
        )
    ]
    for s, files in names:
        s_dir = path.joinpath(s)
        s_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            shutil.copyfile(wav_dir.joinpath(name + ".mp3"), s_dir.joinpath(name + ".mp3"))
            shutil.copyfile(lab_dir.joinpath(name + ".lab"), s_dir.joinpath(name + ".lab"))
    return path


@pytest.fixture()
def basic_corpus_txt_dir(corpus_root_dir, wav_dir, lab_dir):
    path = corpus_root_dir.joinpath("test_basic_txt")
    os.makedirs(path, exist_ok=True)
    names = [("michael", ["acoustic_corpus"]), ("sickmichael", ["cold_corpus", "cold_corpus3"])]
    for s, files in names:
        s_dir = path.joinpath(s)
        os.makedirs(s_dir, exist_ok=True)
        for name in files:
            shutil.copyfile(wav_dir.joinpath(name + ".wav"), s_dir.joinpath(name + ".wav"))
            shutil.copyfile(lab_dir.joinpath(name + ".lab"), s_dir.joinpath(name + ".txt"))
    return path


@pytest.fixture()
def basic_corpus_initial_apostrophe(corpus_root_dir, wav_dir, lab_dir):
    path = corpus_root_dir.joinpath("test_basic_txt")
    os.makedirs(path, exist_ok=True)
    names = [("michael", ["acoustic_corpus"])]
    for s, files in names:
        s_dir = path.joinpath(s)
        os.makedirs(s_dir, exist_ok=True)
        for name in files:
            shutil.copyfile(wav_dir.joinpath(name + ".wav"), s_dir.joinpath(name + ".wav"))
            with open(s_dir.joinpath(name + ".txt"), "w") as outf, open(
                lab_dir.joinpath(name + ".lab"), "r"
            ) as inf:
                outf.write("'" + inf.read())
    return path


@pytest.fixture()
def extra_corpus_dir(corpus_root_dir, wav_dir, lab_dir):
    path = corpus_root_dir.joinpath("test_extra")
    os.makedirs(path, exist_ok=True)
    name = "cold_corpus3"
    shutil.copyfile(wav_dir.joinpath(name + ".wav"), path.joinpath(name + ".wav"))
    shutil.copyfile(lab_dir.joinpath(name + "_extra.lab"), path.joinpath(name + ".lab"))
    return path


@pytest.fixture()
def transcribe_corpus_24bit_dir(corpus_root_dir, wav_dir):
    path = corpus_root_dir.joinpath("test_24bit")
    os.makedirs(path, exist_ok=True)
    name = "cold_corpus_24bit"
    shutil.copyfile(wav_dir.joinpath(name + ".wav"), path.joinpath(name + ".wav"))
    name = "cold_corpus_32bit_float"
    shutil.copyfile(wav_dir.joinpath(name + ".wav"), path.joinpath(name + ".wav"))
    return path


@pytest.fixture()
def stereo_corpus_dir(corpus_root_dir, wav_dir, textgrid_dir):
    path = corpus_root_dir.joinpath("test_stereo")
    os.makedirs(path, exist_ok=True)
    name = "michaelandsickmichael"
    shutil.copyfile(wav_dir.joinpath(name + ".wav"), path.joinpath(name + ".wav"))
    shutil.copyfile(textgrid_dir.joinpath(name + ".TextGrid"), path.joinpath(name + ".TextGrid"))
    return path


@pytest.fixture()
def mp3_corpus_dir(corpus_root_dir, wav_dir, lab_dir):
    path = corpus_root_dir.joinpath("test_cv_mp3")
    os.makedirs(path, exist_ok=True)
    names = ["common_voice_en_22058264", "common_voice_en_22058266", "common_voice_en_22058267"]
    for name in names:
        shutil.copyfile(wav_dir.joinpath(name + ".mp3"), path.joinpath(name + ".mp3"))
        shutil.copyfile(lab_dir.joinpath(name + ".lab"), path.joinpath(name + ".lab"))
    return path


@pytest.fixture()
def opus_corpus_dir(corpus_root_dir, wav_dir, lab_dir):
    path = corpus_root_dir.joinpath("test_mls_opus")
    os.makedirs(path, exist_ok=True)
    names = ["13697_11991_000000"]
    for name in names:
        shutil.copyfile(wav_dir.joinpath(name + ".opus"), path.joinpath(name + ".opus"))
        shutil.copyfile(lab_dir.joinpath(name + ".lab"), path.joinpath(name + ".lab"))
    return path


@pytest.fixture()
def stereo_corpus_short_tg_dir(corpus_root_dir, wav_dir, textgrid_dir):
    path = corpus_root_dir.joinpath("test_stereo_short_tg")
    os.makedirs(path, exist_ok=True)
    name = "michaelandsickmichael"
    shutil.copyfile(wav_dir.joinpath(name + ".wav"), path.joinpath(name + ".wav"))
    shutil.copyfile(
        textgrid_dir.joinpath(name + "_short_tg.TextGrid"),
        path.joinpath(name + ".TextGrid"),
    )
    return path


@pytest.fixture()
def flac_corpus_dir(corpus_root_dir, wav_dir, lab_dir):
    path = corpus_root_dir.joinpath("test_flac_corpus")
    os.makedirs(path, exist_ok=True)
    name = "61-70968-0000"
    shutil.copyfile(wav_dir.joinpath(name + ".flac"), path.joinpath(name + ".flac"))
    shutil.copyfile(lab_dir.joinpath(name + ".lab"), path.joinpath(name + ".lab"))
    return path


@pytest.fixture()
def flac_tg_corpus_dir(corpus_root_dir, wav_dir, textgrid_dir):
    path = corpus_root_dir.joinpath("test_flac_tg_corpus")
    os.makedirs(path, exist_ok=True)
    name = "61-70968-0000"
    shutil.copyfile(wav_dir.joinpath(name + ".flac"), path.joinpath(name + ".flac"))
    shutil.copyfile(textgrid_dir.joinpath(name + ".TextGrid"), path.joinpath(name + ".TextGrid"))
    return path


@pytest.fixture()
def shortsegments_corpus_dir(corpus_root_dir, wav_dir, textgrid_dir):
    path = corpus_root_dir.joinpath("test_short_segments")
    os.makedirs(path, exist_ok=True)
    name = "short_segments"
    shutil.copyfile(wav_dir.joinpath("dummy.wav"), path.joinpath(name + ".wav"))
    shutil.copyfile(textgrid_dir.joinpath(name + ".TextGrid"), path.joinpath(name + ".TextGrid"))
    return path


@pytest.fixture(scope="session")
def dict_dir(test_dir):
    return test_dir.joinpath("dictionaries")


@pytest.fixture(scope="session")
def abstract_dict_path(dict_dir):
    return dict_dir.joinpath("test_abstract.txt")


@pytest.fixture(scope="session")
def basic_dict_path(dict_dir):
    return dict_dir.joinpath("test_basic.txt")


@pytest.fixture(scope="session")
def tabbed_dict_path(dict_dir):
    return dict_dir.joinpath("test_tabbed_dictionary.txt")


@pytest.fixture(scope="session")
def extra_annotations_path(dict_dir):
    return dict_dir.joinpath("test_extra_annotations.txt")


@pytest.fixture(scope="session")
def frclitics_dict_path(dict_dir):
    return dict_dir.joinpath("test_frclitics.txt")


@pytest.fixture(scope="session")
def japanese_dict_path(dict_dir):
    return dict_dir.joinpath("test_japanese.txt")


@pytest.fixture(scope="session")
def hindi_dict_path(dict_dir):
    return dict_dir.joinpath("test_hindi.txt")


@pytest.fixture(scope="session")
def xsampa_dict_path(dict_dir):
    return dict_dir.joinpath("test_xsampa.txt")


@pytest.fixture(scope="session")
def mixed_dict_path(dict_dir):
    return dict_dir.joinpath("test_mixed_format_dictionary.txt")


@pytest.fixture(scope="session")
def english_us_mfa_reduced_dict(dict_dir):
    return dict_dir.joinpath("english_us_mfa_reduced.dict")


@pytest.fixture(scope="session")
def vietnamese_dict_path(dict_dir):
    return dict_dir.joinpath("test_vietnamese_ipa.txt")


@pytest.fixture(scope="session")
def acoustic_dict_path(dict_dir):
    return dict_dir.joinpath("test_acoustic.txt")


@pytest.fixture(scope="session")
def rules_path(config_directory):
    return config_directory.joinpath("test_rules.yaml")


@pytest.fixture(scope="session")
def groups_path(config_directory):
    return config_directory.joinpath("test_groups.yaml")


@pytest.fixture(scope="session")
def mono_output_directory(generated_dir):
    return generated_dir.joinpath("mono_output")


@pytest.fixture(scope="session")
def textgrid_output_model_path(generated_dir):
    return generated_dir.joinpath("textgrid_output_model.zip")


@pytest.fixture(scope="session")
def acoustic_g2p_model_path(generated_dir):
    return generated_dir.joinpath("acoustic_g2p_output_model.zip")


@pytest.fixture(scope="session")
def ivector_output_model_path(generated_dir):
    return generated_dir.joinpath("ivector_output_model.zip")


@pytest.fixture(scope="session")
def basic_g2p_model_path(generated_dir):
    return generated_dir.joinpath("basic_g2p.zip")


@pytest.fixture(scope="session")
def basic_tokenizer_model_path(generated_dir):
    return generated_dir.joinpath("basic_tokenizer.zip")


@pytest.fixture(scope="session")
def basic_phonetisaurus_g2p_model_path(generated_dir):
    return generated_dir.joinpath("basic_phonetisaurus_g2p.zip")


@pytest.fixture(scope="session")
def g2p_basic_output(generated_dir):
    return generated_dir.joinpath("g2p_basic.txt")


@pytest.fixture(scope="session")
def g2p_basic_phonetisaurus_output(generated_dir):
    return generated_dir.joinpath("phonetisaurus_g2p_basic.txt")


@pytest.fixture(scope="session")
def orth_basic_output(generated_dir):
    return generated_dir.joinpath("orth_basic.txt")


@pytest.fixture(scope="session")
def config_directory(test_dir):
    return test_dir.joinpath("configs")


@pytest.fixture(scope="session")
def eval_mapping_path(config_directory):
    return config_directory.joinpath("eval_mapping.yaml")


@pytest.fixture(scope="session")
def basic_train_config_path(config_directory):
    return config_directory.joinpath("basic_train_config.yaml")


@pytest.fixture(scope="session")
def train_g2p_acoustic_config_path(config_directory):
    return config_directory.joinpath("train_g2p_acoustic.yaml")


@pytest.fixture(scope="session")
def transcribe_config_path(config_directory):
    return config_directory.joinpath("transcribe.yaml")


@pytest.fixture(scope="session")
def g2p_config_path(config_directory):
    return config_directory.joinpath("g2p_config.yaml")


@pytest.fixture(scope="session")
def train_g2p_config_path(config_directory):
    return config_directory.joinpath("train_g2p_config.yaml")


@pytest.fixture(scope="session")
def basic_train_lm_config_path(config_directory):
    return config_directory.joinpath("basic_train_lm.yaml")


@pytest.fixture(scope="session")
def different_punctuation_config_path(config_directory):
    return config_directory.joinpath("different_punctuation_config.yaml")


@pytest.fixture(scope="session")
def no_punctuation_config_path(config_directory):
    return config_directory.joinpath("no_punctuation_config.yaml")


@pytest.fixture(scope="session")
def basic_align_config_path(config_directory):
    return config_directory.joinpath("basic_align_config.yaml")


@pytest.fixture(scope="session")
def basic_segment_config_path(config_directory):
    return config_directory.joinpath("basic_segment_config.yaml")


@pytest.fixture(scope="session")
def train_ivector_config_path(config_directory):
    return config_directory.joinpath("ivector_train.yaml")


@pytest.fixture(scope="session")
def mono_align_config_path(config_directory):
    return config_directory.joinpath("mono_align.yaml")


@pytest.fixture(scope="session")
def pron_train_config_path(config_directory):
    return config_directory.joinpath("pron_train.yaml")


@pytest.fixture(scope="session")
def mono_train_config_path(config_directory):
    return config_directory.joinpath("mono_train.yaml")


@pytest.fixture(scope="session")
def xsampa_train_config_path(config_directory):
    return config_directory.joinpath("xsampa_train.yaml")


@pytest.fixture(scope="session")
def tri_train_config_path(config_directory):
    return config_directory.joinpath("tri_train.yaml")


@pytest.fixture(scope="session")
def pitch_train_config_path(config_directory):
    return config_directory.joinpath("pitch_tri_train.yaml")


@pytest.fixture(scope="session")
def lda_train_config_path(config_directory):
    return config_directory.joinpath("lda_train.yaml")


@pytest.fixture(scope="session")
def sat_train_config_path(config_directory):
    return config_directory.joinpath("sat_train.yaml")


@pytest.fixture(scope="session")
def multispeaker_dictionary_config_path(generated_dir, basic_dict_path, english_dictionary):
    path = generated_dir.joinpath("multispeaker_dictionary.yaml")
    with mfa_open(path, "w") as f:
        yaml.dump(
            {"default": english_dictionary, "michael": basic_dict_path},
            f,
            Dumper=yaml.Dumper,
            allow_unicode=True,
        )
    return path


@pytest.fixture(scope="session")
def mfa_speaker_dict_path(generated_dir, english_uk_mfa_dictionary, english_us_mfa_reduced_dict):
    path = generated_dir.joinpath("test_multispeaker_mfa_dictionary.yaml")
    with mfa_open(path, "w") as f:
        yaml.dump(
            {"default": english_us_mfa_reduced_dict, "speaker": english_us_mfa_reduced_dict},
            f,
            Dumper=yaml.Dumper,
            allow_unicode=True,
        )
    return path


@pytest.fixture(scope="session")
def english_mfa_phone_groups_path(config_directory):
    path = config_directory.joinpath("acoustic", "english_mfa_phone_groups.yaml")
    return path


@pytest.fixture(scope="session")
def english_mfa_rules_path(config_directory):
    path = config_directory.joinpath("acoustic", "english_mfa_rules.yaml")
    return path


@pytest.fixture(scope="session")
def english_mfa_topology_path(config_directory):
    path = config_directory.joinpath("acoustic", "english_mfa_topology.yaml")
    return path


@pytest.fixture(scope="session")
def bad_topology_path(config_directory):
    path = config_directory.joinpath("acoustic", "bad_topology.yaml")
    return path


@pytest.fixture(scope="session")
def test_align_config():
    return {"beam": 100, "retry_beam": 400}


@pytest.fixture(scope="session")
def reference_transcripts():
    return {
        "mfa_cutoff": "<cutoff-montreal> montreal <cutoff-montreal> <cutoff-montreal> montreal <cutoff-forced> <cutoff-forced> forced <cutoff-aligner> aligner <cutoff-aligner> aligner",
        "mfa_whatscalled": "montreal forced what's called aligner",
        "mfa_uhuh": "montreal uh uh uh uh uh uh forced aligner",
        "mfa_uhum": "montreal forced uh um uh hm hm um forced aligner",
        "mfa_michael": "montreal forced aligner",
        "mfa_kmg": "montreal forced aligner",
        "mfa_falsetto": "montreal forced aligner",
        "mfa_whisper": "montreal forced aligner",
        "mfa_exaggerated": "montreal forced aligner",
        "mfa_breathy": "montreal forced aligner",
        "mfa_creaky": "montreal forced aligner",
        "mfa_long": "montreal forced aligner",
        "mfa_hes": "montreal <hes-forced> aligner",
        "mfa_longstop": "this is a long stop",
        "mfa_putty": "m f a is like putty",
        "mfa_puddy": "m f a is like puddy",
        "mfa_puttynorm": "m f a is like putty",
        "mfa_pooty": "m f a is like pooty",
        "mfa_bottle": "m f a is like bottle",
        "mfa_patty": "m f a is like patty",
        "mfa_buddy": "m f a is like buddy",
        "mfa_apex": "m f a is like apex",
        "mfa_poofy": "m f a is like poofy",
        "mfa_reallylong": "m f a is like this is so many words right here",
        "mfa_internalsil": "<hes-montreal> <hes-forced> <hes-aligner>",
        "mfa_her": "montreal forced aligner i hardly know her",
        "mfa_er": "montreal forced aligner i hardly know 'er",
        "mfa_erpause": "montreal forced aligner i hardly know 'er",
        "mfa_cutoffprogressive": "<cutoff-montreal> <cutoff-montreal> uh <cutoff-montreal> montreal <cutoff-forced> forced <cutoff-aligner> <cutoff-aligner> hm <cutoff-aligner> <cutoff-aligner> aligner aligner",
        "mfa_affectation": "montreal forced aligner",
        "mfa_crossword": "but um montreal but um but montreal forced aligner",
        "mfa_registershift": "montreal forced forced aligner",
        "falsetto": "this is all very high pitched",
        "falsetto2": "i really don't know how people talk like this",
        "whisper": "this is all very whispered",
        "whisper2": "there's gonna be no voiced speech whatsoever here",
        "mfa_uh": "montreal uh forced aligner",
        "mfa_um": "montreal forced um aligner",
        "mfa_youknow": "you know montreal forced aligner",
        "mfa_unk": "montreal forced <unk> aligner",
        "mfa_words": "montreal forced aligner word another word here's some more word word word word word",
        "mfa_surround": "this one montreal is going to be forced very bad aligner but what are you gonna do",
        "mfa_breaths": "montreal forced aligner",
        "mfa_laughter": "[laughter] montreal [laughter] forced [laughter] aligner [laughter]",
        "mfa_the": "the montreal forced aligner",
        "mfa_thenorm": "the montreal forced aligner",
        "mfa_thestop": "this is the montreal forced aligner",
        "mfa_theapprox": "this is the montreal forced aligner",
        "mfa_thez": "this is the montreal forced aligner",
        "mfa_thea": "this is a montreal forced aligner",
        "mfa_theinitialstop": "the montreal forced aligner",
        "mfa_theother": "this is the other montreal forced aligner",
        "mfa_thoughts": "i have a thousand thoughts about that thing",
    }


@pytest.fixture(scope="session")
def filler_insertion_utterances():
    return [
        "mfa_michael",
        "mfa_uh",
        "mfa_um",
        "mfa_youknow",
        "mfa_unk",
        "mfa_words",
        "mfa_surround",
        "mfa_breaths",
        "mfa_laughter",
        "mfa_cutoffprogressive",
        "mfa_uhuh",
        "mfa_uhum",
        "mfa_whatscalled",
        "mfa_cutoff",
        "mfa_exaggerated",
    ]


@pytest.fixture(scope="session")
def putty_utterances():
    return [
        "mfa_putty",
        "mfa_puddy",
        "mfa_puttynorm",
        "mfa_pooty",
        "mfa_bottle",
        "mfa_patty",
        "mfa_buddy",
        "mfa_apex",
        "mfa_poofy",
        "mfa_reallylong",
    ]


@pytest.fixture(scope="session")
def register_utterances():
    return [
        "mfa_michael",
        "mfa_kmg",
        "mfa_falsetto",
        "mfa_whisper",
        "mfa_exaggerated",
        "mfa_breathy",
        "mfa_creaky",
        "mfa_registershift",
        "falsetto",
        "falsetto2",
        "whisper",
        "whisper2",
    ]


@pytest.fixture(scope="session")
def pronunciation_variation_utterances():
    return [
        "mfa_crossword",
        "mfa_her",
        "mfa_er",
        "mfa_erpause",
        "mfa_the",
        "mfa_thenorm",
        "mfa_thestop",
        "mfa_theapprox",
        "mfa_thez",
        "mfa_theinitialstop",
        "mfa_theother",
        "mfa_thoughts",
    ]


@pytest.fixture(scope="session")
def cutoff_utterances():
    return [
        "mfa_cutoff",
        "mfa_cutoffprogressive",
        "mfa_internalsil",
        "mfa_longstop",
        "mfa_long",
        "mfa_hes",
    ]


@pytest.fixture(scope="session")
def filler_insertion_corpus(filler_insertion_utterances, corpus_root_dir, wav_dir, lab_dir):
    path = corpus_root_dir.joinpath("test_filler_insertion")
    path.mkdir(exist_ok=True, parents=True)
    speaker_name = "michael"
    s_dir = path.joinpath(speaker_name)
    s_dir.mkdir(exist_ok=True, parents=True)
    transcript = "montreal forced aligner"
    for name in filler_insertion_utterances:
        shutil.copyfile(wav_dir.joinpath(name + ".flac"), s_dir.joinpath(name + ".flac"))
        with mfa_open(s_dir.joinpath(name + ".lab"), "w") as f:
            f.write(transcript)
    return path


@pytest.fixture(scope="session")
def pronunciation_variation_corpus(
    pronunciation_variation_utterances, corpus_root_dir, wav_dir, lab_dir, reference_transcripts
):
    path = corpus_root_dir.joinpath("test_pronunciation_variation")
    path.mkdir(exist_ok=True, parents=True)
    speaker_name = "michael"
    s_dir = path.joinpath(speaker_name)
    s_dir.mkdir(exist_ok=True, parents=True)
    for name in pronunciation_variation_utterances:
        shutil.copyfile(wav_dir.joinpath(name + ".flac"), s_dir.joinpath(name + ".flac"))
        with mfa_open(s_dir.joinpath(name + ".lab"), "w") as f:
            f.write(reference_transcripts[name])
    return path
