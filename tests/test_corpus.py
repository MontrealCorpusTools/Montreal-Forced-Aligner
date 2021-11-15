import os
import shutil

import pytest

from montreal_forced_aligner.config.train_config import train_yaml_to_config
from montreal_forced_aligner.corpus import Corpus
from montreal_forced_aligner.corpus.classes import File, Speaker, Utterance
from montreal_forced_aligner.corpus.helper import get_wav_info
from montreal_forced_aligner.dictionary import MultispeakerDictionary
from montreal_forced_aligner.exceptions import SoxError


def test_mp3(mp3_test_path):
    try:
        info = get_wav_info(mp3_test_path)
        assert "sox_string" in info
    except SoxError:
        pytest.skip()


def test_add(basic_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, "basic")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    c = Corpus(basic_corpus_dir, output_directory, use_mp=True)
    new_speaker = Speaker("new_speaker")
    new_file = File("new_file.wav", "new_file.txt")
    new_utterance = Utterance(new_speaker, new_file, text="blah blah")
    utterance_id = new_utterance.name
    assert utterance_id not in c.utterances
    c.add_utterance(new_utterance)

    assert utterance_id in c.utterances
    assert utterance_id in c.speakers["new_speaker"].utterances
    assert utterance_id in c.files["new_file"].utterances
    assert c.utterances[utterance_id].text == "blah blah"

    c.delete_utterance(utterance_id)
    assert utterance_id not in c.utterances
    assert "new_speaker" in c.speakers
    assert "new_file" in c.files


def test_basic(basic_dict_path, basic_corpus_dir, generated_dir, default_feature_config):
    output_directory = os.path.join(generated_dir, "basic")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = MultispeakerDictionary(basic_dict_path, output_directory)
    dictionary.write()
    c = Corpus(basic_corpus_dir, output_directory, use_mp=True)
    c.initialize_corpus(dictionary, default_feature_config)
    for speaker in c.speakers.values():
        data = speaker.dictionary.data()
        assert speaker.dictionary.config.silence_phones == data.dictionary_config.silence_phones
        assert (
            speaker.dictionary.config.multilingual_ipa == data.dictionary_config.multilingual_ipa
        )
        assert speaker.dictionary.words_mapping == data.words_mapping
        assert speaker.dictionary.config.punctuation == data.dictionary_config.punctuation
        assert speaker.dictionary.config.clitic_markers == data.dictionary_config.clitic_markers
        assert speaker.dictionary.oov_int == data.oov_int
        assert speaker.dictionary.words == data.words
    assert c.get_feat_dim() == 39


def test_basic_txt(basic_corpus_txt_dir, basic_dict_path, generated_dir, default_feature_config):
    output_directory = os.path.join(generated_dir, "basic")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = MultispeakerDictionary(basic_dict_path, os.path.join(generated_dir, "basic"))
    dictionary.write()
    c = Corpus(basic_corpus_txt_dir, output_directory, use_mp=False)
    print(c.no_transcription_files)
    assert len(c.no_transcription_files) == 0
    c.initialize_corpus(dictionary, default_feature_config)
    assert c.get_feat_dim() == 39


def test_alignable_from_temp(
    basic_corpus_txt_dir, basic_dict_path, generated_dir, default_feature_config
):
    dictionary = MultispeakerDictionary(basic_dict_path, os.path.join(generated_dir, "basic"))
    dictionary.write()
    output_directory = os.path.join(generated_dir, "basic")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    c = Corpus(basic_corpus_txt_dir, output_directory, use_mp=False)
    assert len(c.no_transcription_files) == 0
    c.initialize_corpus(dictionary, default_feature_config)
    assert c.get_feat_dim() == 39

    c = Corpus(basic_corpus_txt_dir, output_directory, use_mp=False)
    assert len(c.no_transcription_files) == 0
    c.initialize_corpus(dictionary, default_feature_config)
    assert c.get_feat_dim() == 39


def test_transcribe_from_temp(
    basic_corpus_txt_dir, basic_dict_path, generated_dir, default_feature_config
):
    dictionary = MultispeakerDictionary(basic_dict_path, os.path.join(generated_dir, "basic"))
    dictionary.write()
    output_directory = os.path.join(generated_dir, "basic")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    c = Corpus(basic_corpus_txt_dir, output_directory, use_mp=False)
    c.initialize_corpus(dictionary, default_feature_config)
    assert c.get_feat_dim() == 39

    c = Corpus(basic_corpus_txt_dir, output_directory, use_mp=False)
    c.initialize_corpus(dictionary, default_feature_config)
    assert c.get_feat_dim() == 39


def test_extra(sick_dict, extra_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, "extra")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    corpus = Corpus(extra_corpus_dir, output_directory, num_jobs=2, use_mp=False)
    corpus.initialize_corpus(sick_dict)


def test_stereo(basic_dict_path, stereo_corpus_dir, temp_dir, default_feature_config):
    temp = os.path.join(temp_dir, "stereo")
    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = MultispeakerDictionary(basic_dict_path, os.path.join(temp, "basic"))
    dictionary.write()
    d = Corpus(stereo_corpus_dir, temp, use_mp=False)
    d.initialize_corpus(dictionary, default_feature_config)
    assert d.get_feat_dim() == 39


def test_stereo_short_tg(
    basic_dict_path, stereo_corpus_short_tg_dir, temp_dir, default_feature_config
):
    temp = os.path.join(temp_dir, "stereo_tg")
    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = MultispeakerDictionary(basic_dict_path, os.path.join(temp, "basic"))
    dictionary.write()
    d = Corpus(stereo_corpus_short_tg_dir, temp, use_mp=False)
    d.initialize_corpus(dictionary, default_feature_config)
    assert d.get_feat_dim() == 39


def test_flac(basic_dict_path, flac_corpus_dir, temp_dir, default_feature_config):
    temp = os.path.join(temp_dir, "flac")
    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = MultispeakerDictionary(basic_dict_path, os.path.join(temp, "basic"))
    dictionary.write()
    d = Corpus(flac_corpus_dir, temp, use_mp=False)
    d.initialize_corpus(dictionary, default_feature_config)
    assert d.get_feat_dim() == 39


def test_audio_directory(basic_dict_path, basic_split_dir, temp_dir, default_feature_config):
    temp = os.path.join(temp_dir, "audio_dir_test")
    audio_dir, text_dir = basic_split_dir
    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = MultispeakerDictionary(basic_dict_path, os.path.join(temp, "basic"))
    dictionary.write()
    d = Corpus(text_dir, temp, use_mp=False, audio_directory=audio_dir)
    assert len(d.no_transcription_files) == 0
    assert len(d.files) > 0
    d.initialize_corpus(dictionary, default_feature_config)
    assert d.get_feat_dim() == 39

    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = MultispeakerDictionary(basic_dict_path, os.path.join(temp, "basic"))
    dictionary.write()
    d = Corpus(text_dir, temp, use_mp=True, audio_directory=audio_dir)
    assert len(d.no_transcription_files) == 0
    assert len(d.files) > 0
    d.initialize_corpus(dictionary, default_feature_config)
    assert d.get_feat_dim() == 39


def test_flac_mp(basic_dict_path, flac_corpus_dir, temp_dir, default_feature_config):
    temp = os.path.join(temp_dir, "flac")
    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = MultispeakerDictionary(basic_dict_path, os.path.join(temp, "basic"))
    dictionary.write()
    d = Corpus(flac_corpus_dir, temp, use_mp=True)
    d.initialize_corpus(dictionary, default_feature_config)
    assert d.get_feat_dim() == 39


def test_flac_tg(basic_dict_path, flac_tg_corpus_dir, temp_dir, default_feature_config):
    temp = os.path.join(temp_dir, "flac")
    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = MultispeakerDictionary(basic_dict_path, os.path.join(temp, "basic"))
    dictionary.write()
    d = Corpus(flac_tg_corpus_dir, temp, use_mp=False)
    d.initialize_corpus(dictionary, default_feature_config)
    assert d.get_feat_dim() == 39


def test_flac_tg_mp(basic_dict_path, flac_tg_corpus_dir, temp_dir, default_feature_config):
    temp = os.path.join(temp_dir, "flac")
    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = MultispeakerDictionary(basic_dict_path, os.path.join(temp, "basic"))
    dictionary.write()
    d = Corpus(flac_tg_corpus_dir, temp, use_mp=True)
    d.initialize_corpus(dictionary, default_feature_config)
    assert d.get_feat_dim() == 39


def test_flac_tg_transcribe(basic_dict_path, flac_tg_corpus_dir, temp_dir, default_feature_config):
    temp = os.path.join(temp_dir, "flac_tg")
    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = MultispeakerDictionary(basic_dict_path, os.path.join(temp, "basic"))
    dictionary.write()
    d = Corpus(flac_tg_corpus_dir, temp, use_mp=False)
    d.initialize_corpus(dictionary, default_feature_config)
    assert d.get_feat_dim() == 39

    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = MultispeakerDictionary(basic_dict_path, os.path.join(temp, "basic"))
    dictionary.write()
    d = Corpus(flac_tg_corpus_dir, temp, use_mp=True)
    d.initialize_corpus(dictionary, default_feature_config)
    assert d.get_feat_dim() == 39


def test_flac_transcribe(
    basic_dict_path, flac_transcribe_corpus_dir, temp_dir, default_feature_config
):
    temp = os.path.join(temp_dir, "flac_transcribe")
    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = MultispeakerDictionary(basic_dict_path, os.path.join(temp, "basic"))
    dictionary.write()
    d = Corpus(flac_transcribe_corpus_dir, temp, use_mp=True)
    d.initialize_corpus(dictionary, default_feature_config)
    assert d.get_feat_dim() == 39

    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = MultispeakerDictionary(basic_dict_path, os.path.join(temp, "basic"))
    dictionary.write()

    d = Corpus(flac_transcribe_corpus_dir, temp, use_mp=False)
    d.initialize_corpus(dictionary, default_feature_config)
    assert d.get_feat_dim() == 39


def test_24bit_wav(transcribe_corpus_24bit_dir, temp_dir, default_feature_config):
    temp = os.path.join(temp_dir, "24bit")
    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)

    c = Corpus(transcribe_corpus_24bit_dir, temp, use_mp=False)
    c.initialize_corpus(feature_config=default_feature_config)
    assert c.get_feat_dim() == 39
    assert len(c.files) == 2


def test_short_segments(
    basic_dict_path, shortsegments_corpus_dir, temp_dir, default_feature_config
):
    temp = os.path.join(temp_dir, "short_segments")
    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = MultispeakerDictionary(basic_dict_path, temp)
    dictionary.write()
    corpus = Corpus(shortsegments_corpus_dir, temp, use_mp=False)
    corpus.initialize_corpus(dictionary, default_feature_config)
    assert len(corpus.utterances) == 3
    assert len([x for x in corpus.utterances.values() if not x.ignored]) == 1
    assert len([x for x in corpus.utterances.values() if x.features is not None]) == 1
    assert len([x for x in corpus.utterances.values() if x.ignored]) == 2
    assert len([x for x in corpus.utterances.values() if x.features is None]) == 2


def test_speaker_groupings(
    multilingual_ipa_corpus_dir, temp_dir, english_us_ipa_dictionary, default_feature_config
):
    output_directory = os.path.join(temp_dir, "speaker_groupings")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = MultispeakerDictionary(english_us_ipa_dictionary, output_directory)
    dictionary.write()
    c = Corpus(multilingual_ipa_corpus_dir, output_directory, use_mp=True)

    c.initialize_corpus(dictionary, default_feature_config)
    speakers = os.listdir(multilingual_ipa_corpus_dir)
    for s in speakers:
        assert any(s in x.speakers for x in c.jobs)
    for _, _, files in os.walk(multilingual_ipa_corpus_dir):
        for f in files:
            name, ext = os.path.splitext(f)
            assert name in c.files

    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary.write()
    c = Corpus(multilingual_ipa_corpus_dir, output_directory, num_jobs=1, use_mp=True)

    c.initialize_corpus(dictionary, default_feature_config)
    for s in speakers:
        assert any(s in x.speakers for x in c.jobs)
    for _, _, files in os.walk(multilingual_ipa_corpus_dir):
        for f in files:
            name, ext = os.path.splitext(f)
            assert name in c.files


def test_subset(
    multilingual_ipa_corpus_dir, temp_dir, english_us_ipa_dictionary, default_feature_config
):
    output_directory = os.path.join(temp_dir, "large_subset")
    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = MultispeakerDictionary(english_us_ipa_dictionary, output_directory)
    dictionary.write()
    c = Corpus(multilingual_ipa_corpus_dir, output_directory, use_mp=False)
    c.initialize_corpus(dictionary, default_feature_config)
    sd = c.split_directory

    s = c.subset_directory(5)
    assert os.path.exists(sd)
    assert os.path.exists(s)


def test_weird_words(weird_words_dir, temp_dir, sick_dict_path):
    output_directory = os.path.join(temp_dir, "weird_words")
    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = MultispeakerDictionary(sick_dict_path, output_directory)
    assert "i’m" not in dictionary.default_dictionary.words
    assert "’m" not in dictionary.default_dictionary.words
    assert dictionary.default_dictionary.words["i'm"][0]["pronunciation"] == ("ay", "m", "ih")
    assert dictionary.default_dictionary.words["i'm"][1]["pronunciation"] == ("ay", "m")
    assert dictionary.default_dictionary.words["'m"][0]["pronunciation"] == ("m",)
    dictionary.write()
    c = Corpus(weird_words_dir, output_directory, use_mp=False)
    c.initialize_corpus(dictionary)
    assert c.utterances["weird-words-weird-words"].oovs == [
        "talking-ajfish",
        "asds-asda",
        "sdasd-me",
    ]

    dictionary.set_word_set(c.word_set)
    for w in ["i'm", "this'm", "sdsdsds'm", "'m"]:
        _ = dictionary.default_dictionary.to_int(w)
    print(dictionary.oovs_found)
    assert "'m" not in dictionary.oovs_found


def test_punctuated(punctuated_dir, temp_dir, sick_dict_path):
    output_directory = os.path.join(temp_dir, "punctuated")
    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = MultispeakerDictionary(sick_dict_path, output_directory)
    dictionary.write()
    c = Corpus(punctuated_dir, output_directory, dictionary_config=dictionary.config, use_mp=False)
    c.initialize_corpus(dictionary)
    assert (
        c.utterances["punctuated-punctuated"].text
        == "oh yes they they you know they love her and so i mean"
    )


def test_alternate_punctuation(
    punctuated_dir, temp_dir, sick_dict_path, different_punctuation_config
):
    train_config, align_config, dictionary_config = train_yaml_to_config(
        different_punctuation_config
    )
    output_directory = os.path.join(temp_dir, "punctuated")
    shutil.rmtree(output_directory, ignore_errors=True)
    print(dictionary_config.punctuation)
    dictionary = MultispeakerDictionary(sick_dict_path, output_directory, dictionary_config)
    dictionary.write()
    c = Corpus(
        punctuated_dir,
        output_directory,
        dictionary_config,
        use_mp=False,
    )
    c.initialize_corpus(dictionary)
    assert (
        c.utterances["punctuated-punctuated"].text
        == "oh yes, they they, you know, they love her and so i mean"
    )


def test_xsampa_corpus(
    xsampa_corpus_dir, xsampa_dict_path, temp_dir, generated_dir, different_punctuation_config
):
    train_config, align_config, dictionary_config = train_yaml_to_config(
        different_punctuation_config
    )
    output_directory = os.path.join(temp_dir, "xsampa")
    shutil.rmtree(output_directory, ignore_errors=True)
    print(dictionary_config.punctuation)
    dictionary = MultispeakerDictionary(xsampa_dict_path, output_directory, dictionary_config)
    dictionary.write()
    c = Corpus(xsampa_corpus_dir, output_directory, dictionary_config, use_mp=False)
    c.initialize_corpus(dictionary)
    assert (
        c.utterances["xsampa-michael"].text
        == r"@bUr\tOU {bstr\{kt {bSaIr\ Abr\utseIzi {br\@geItIN @bor\n {b3kr\Ambi {bI5s@`n Ar\g thr\Ip@5eI Ar\dvAr\k".lower()
    )
