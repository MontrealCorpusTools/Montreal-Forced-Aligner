import os
import shutil

import pytest

from montreal_forced_aligner.corpus.acoustic_corpus import (
    AcousticCorpus,
    AcousticCorpusWithPronunciations,
)
from montreal_forced_aligner.corpus.classes import File, Speaker, Utterance
from montreal_forced_aligner.corpus.helper import get_wav_info
from montreal_forced_aligner.corpus.text_corpus import TextCorpus
from montreal_forced_aligner.dictionary.pronunciation import Pronunciation
from montreal_forced_aligner.exceptions import SoxError


def test_mp3(mp3_test_path):
    try:
        info = get_wav_info(mp3_test_path)
        assert info.sox_string
    except SoxError:
        pytest.skip()


def test_speaker_word_set(
    multilingual_ipa_tg_corpus_dir, multispeaker_dictionary_config_path, temp_dir
):
    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=multilingual_ipa_tg_corpus_dir,
        dictionary_path=multispeaker_dictionary_config_path,
        temporary_directory=temp_dir,
    )
    corpus.load_corpus()
    speaker_one = corpus.speakers["speaker_one"]
    assert speaker_one.dictionary.lookup("chad-like") == ["chad", "like"]
    assert speaker_one.dictionary.oov_int not in speaker_one.dictionary.to_int("chad-like")


def test_add(basic_corpus_dir, sick_dict_path, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    corpus = AcousticCorpus(
        corpus_directory=basic_corpus_dir,
        use_mp=True,
        temporary_directory=output_directory,
    )
    corpus._load_corpus()
    new_speaker = Speaker("new_speaker")
    new_file = File(
        os.path.join(basic_corpus_dir, "michael", "acoustic_corpus.wav"),
        os.path.join(basic_corpus_dir, "michael", "acoustic_corpus.lab"),
        name="new_file",
    )
    new_utterance = Utterance(new_speaker, new_file, text="blah blah")
    utterance_id = new_utterance.name
    assert utterance_id not in corpus.utterances
    corpus.add_utterance(new_utterance)

    assert utterance_id in corpus.utterances
    assert utterance_id in corpus.speakers["new_speaker"].utterances
    assert utterance_id in corpus.files["new_file"].utterances
    assert corpus.utterances[utterance_id].text == "blah blah"

    corpus.delete_utterance(utterance_id)
    assert utterance_id not in corpus.utterances
    assert "new_speaker" in corpus.speakers
    assert "new_file" in corpus.files


def test_basic(basic_dict_path, basic_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=basic_corpus_dir,
        dictionary_path=basic_dict_path,
        use_mp=False,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    for speaker in corpus.speakers:
        assert speaker.dictionary.silence_phones == corpus.silence_phones
        assert (
            speaker.dictionary.words_mapping == corpus.get_dictionary(speaker.name).words_mapping
        )
        assert speaker.dictionary.punctuation == corpus.punctuation
        assert speaker.dictionary.clitic_markers == corpus.clitic_markers
        assert speaker.dictionary.oov_int == corpus.get_dictionary(speaker.name).oov_int
        assert speaker.dictionary.words == corpus.get_dictionary(speaker.name).words
    assert corpus.get_feat_dim() == 39


def test_basic_txt(basic_corpus_txt_dir, basic_dict_path, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    corpus = AcousticCorpus(
        corpus_directory=basic_corpus_txt_dir,
        use_mp=False,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()

    print(corpus.no_transcription_files)
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39


def test_acoustic_from_temp(basic_corpus_txt_dir, basic_dict_path, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    corpus = AcousticCorpus(
        corpus_directory=basic_corpus_txt_dir,
        use_mp=False,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39

    new_corpus = AcousticCorpus(
        corpus_directory=basic_corpus_txt_dir,
        use_mp=False,
        temporary_directory=output_directory,
    )
    new_corpus.load_corpus()
    assert len(new_corpus.no_transcription_files) == 0
    assert new_corpus.get_feat_dim() == 39


def test_text_corpus_from_temp(basic_corpus_txt_dir, basic_dict_path, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    corpus = TextCorpus(
        corpus_directory=basic_corpus_txt_dir,
        dictionary_path=basic_dict_path,
        use_mp=False,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.utterances) > 0


def test_extra(sick_dict, extra_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=extra_corpus_dir,
        dictionary_path=sick_dict,
        use_mp=False,
        num_jobs=2,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39


def test_stereo(basic_dict_path, stereo_corpus_dir, generated_dir):

    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=stereo_corpus_dir,
        use_mp=False,
        num_jobs=1,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39
    assert corpus.files["michaelandsickmichael"].num_channels == 2


def test_stereo_short_tg(basic_dict_path, stereo_corpus_short_tg_dir, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=stereo_corpus_short_tg_dir,
        use_mp=False,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39
    assert corpus.files["michaelandsickmichael"].num_channels == 2


def test_flac(basic_dict_path, flac_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=flac_corpus_dir,
        use_mp=False,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39


def test_audio_directory(basic_dict_path, basic_split_dir, generated_dir):
    audio_dir, text_dir = basic_split_dir
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=text_dir,
        use_mp=False,
        audio_directory=audio_dir,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39
    assert len(corpus.files) > 0

    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    corpus = AcousticCorpus(
        corpus_directory=text_dir,
        use_mp=True,
        audio_directory=audio_dir,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39
    assert len(corpus.files) > 0


def test_flac_mp(basic_dict_path, flac_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=flac_corpus_dir,
        use_mp=True,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39
    assert len(corpus.files) > 0


def test_flac_tg(basic_dict_path, flac_tg_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=flac_tg_corpus_dir,
        use_mp=False,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39
    assert len(corpus.files) > 0


def test_flac_tg_mp(basic_dict_path, flac_tg_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=flac_tg_corpus_dir,
        use_mp=True,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39
    assert len(corpus.files) > 0


def test_24bit_wav(transcribe_corpus_24bit_dir, basic_dict_path, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=transcribe_corpus_24bit_dir,
        use_mp=False,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 2
    assert corpus.get_feat_dim() == 39
    assert len(corpus.files) > 0


def test_short_segments(shortsegments_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=shortsegments_corpus_dir,
        use_mp=False,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.utterances._data) == 3
    assert len([x for x in corpus.utterances._data.values() if not x.ignored]) == 1
    assert len([x for x in corpus.utterances._data.values() if x.features is not None]) == 2
    assert len([x for x in corpus.utterances._data.values() if x.ignored]) == 2
    assert len([x for x in corpus.utterances._data.values() if x.features is None]) == 1


def test_speaker_groupings(multilingual_ipa_corpus_dir, generated_dir, english_us_ipa_dictionary):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=multilingual_ipa_corpus_dir,
        dictionary_path=english_us_ipa_dictionary,
        use_mp=True,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    speakers = os.listdir(multilingual_ipa_corpus_dir)
    for s in speakers:
        assert any(s in x.speakers for x in corpus.jobs)
    for _, _, files in os.walk(multilingual_ipa_corpus_dir):
        for f in files:
            name, ext = os.path.splitext(f)
            for file in corpus.files:
                if name == file._name:
                    break
            else:
                raise Exception(f"File {name} not loaded")

    shutil.rmtree(output_directory, ignore_errors=True)
    new_corpus = AcousticCorpusWithPronunciations(
        corpus_directory=multilingual_ipa_corpus_dir,
        dictionary_path=english_us_ipa_dictionary,
        num_jobs=1,
        use_mp=True,
        temporary_directory=output_directory,
    )
    new_corpus.load_corpus()
    for s in speakers:
        assert any(s in x.speakers for x in new_corpus.jobs)
    for _, _, files in os.walk(multilingual_ipa_corpus_dir):
        for f in files:
            name, ext = os.path.splitext(f)
            for file in new_corpus.files:
                if name == file._name:
                    break
            else:
                raise Exception(f"File {name} not loaded")


def test_subset(multilingual_ipa_corpus_dir, generated_dir, english_us_ipa_dictionary):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=multilingual_ipa_corpus_dir,
        use_mp=True,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    sd = corpus.split_directory

    s = corpus.subset_directory(5)
    assert os.path.exists(sd)
    assert os.path.exists(s)


def test_weird_words(weird_words_dir, generated_dir, sick_dict_path):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=weird_words_dir,
        dictionary_path=sick_dict_path,
        use_mp=True,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert "i’m" not in corpus.default_dictionary.words
    assert "’m" not in corpus.default_dictionary.words
    assert (
        Pronunciation(("ay", "m", "ih"), 1, None, None, None)
        in corpus.default_dictionary.words["i'm"].pronunciations
    )
    assert (
        Pronunciation(("ay", "m"), 1, None, None, None)
        in corpus.default_dictionary.words["i'm"].pronunciations
    )
    assert (
        Pronunciation(("m",), 1, None, None, None)
        in corpus.default_dictionary.words["'m"].pronunciations
    )
    print(corpus.utterances)
    assert corpus.utterances["weird-words-weird-words"].oovs == {
        "ajfish",
        "asds-asda",
        "sdasd",
        "<s>",
        "</s>",
    }
    print(corpus.utterances["weird-words-weird-words"].text_int_for_scp())
    assert (
        corpus.utterances["weird-words-weird-words"].text_int_for_scp()[-1]
        == corpus.default_dictionary.oov_int
    )
    corpus.set_lexicon_word_set(corpus.corpus_word_set)
    for w in ["i'm", "this'm", "sdsdsds'm", "'m"]:
        _ = corpus.default_dictionary.to_int(w)
    print(corpus.oovs_found)
    assert "'m" not in corpus.oovs_found


def test_punctuated(punctuated_dir, generated_dir, sick_dict_path):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=punctuated_dir,
        dictionary_path=sick_dict_path,
        use_mp=True,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    print(corpus.utterances)
    assert (
        corpus.utterances["punctuated-punctuated"].text
        == "oh yes they they you know they love her and so i mean"
    )


def test_alternate_punctuation(
    punctuated_dir, generated_dir, sick_dict_path, different_punctuation_config_path
):
    from montreal_forced_aligner.acoustic_modeling.trainer import TrainableAligner

    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    params, skipped = AcousticCorpusWithPronunciations.extract_relevant_parameters(
        TrainableAligner.parse_parameters(different_punctuation_config_path)
    )
    params["use_mp"] = True
    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=punctuated_dir,
        dictionary_path=sick_dict_path,
        temporary_directory=output_directory,
        **params,
    )
    corpus.load_corpus()
    assert (
        corpus.utterances["punctuated-punctuated"].text
        == "oh yes, they they, you know, they love her and so i mean"
    )


def test_no_punctuation(punctuated_dir, generated_dir, sick_dict_path, no_punctuation_config_path):
    from montreal_forced_aligner.acoustic_modeling.trainer import TrainableAligner

    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    params, skipped = AcousticCorpusWithPronunciations.extract_relevant_parameters(
        TrainableAligner.parse_parameters(no_punctuation_config_path)
    )
    params["use_mp"] = True
    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=punctuated_dir,
        dictionary_path=sick_dict_path,
        temporary_directory=output_directory,
        **params,
    )
    assert not corpus.punctuation
    assert not corpus.compound_markers
    assert not corpus.clitic_markers
    corpus.load_corpus()
    punctuated = corpus.utterances["punctuated-punctuated"]
    assert punctuated.text == "oh yes, they - they, you know, they love her and so i mean..."
    assert punctuated.normalized_text == [
        "oh",
        "yes,",
        "they",
        "-",
        "they,",
        "you",
        "know,",
        "they",
        "love",
        "her",
        "and",
        "so",
        "i",
        "mean...",
    ]
    weird_words = corpus.utterances["punctuated-weird-words"]
    assert weird_words.text == "i’m talking-ajfish me-really asds-asda sdasd-me <s> </s>"
    assert weird_words.normalized_text == [
        "i’m",
        "talking-ajfish",
        "me-really",
        "asds-asda",
        "sdasd-me",
        "<s>",
        "</s>",
    ]


def test_xsampa_corpus(
    xsampa_corpus_dir, xsampa_dict_path, generated_dir, different_punctuation_config_path
):
    from montreal_forced_aligner.acoustic_modeling.trainer import TrainableAligner

    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    params, skipped = AcousticCorpusWithPronunciations.extract_relevant_parameters(
        TrainableAligner.parse_parameters(different_punctuation_config_path)
    )
    params["use_mp"] = True
    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=xsampa_corpus_dir,
        dictionary_path=xsampa_dict_path,
        temporary_directory=output_directory,
        **params,
    )
    corpus.load_corpus()
    assert (
        corpus.utterances["michael-xsampa"].text
        == r"@bUr\tOU {bstr\{kt {bSaIr\ Abr\utseIzi {br\@geItIN @bor\n {b3kr\Ambi {bI5s@`n Ar\g thr\Ip@5eI Ar\dvAr\k".lower()
    )
