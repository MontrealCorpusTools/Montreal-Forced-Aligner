import os
import shutil
import pytest

from montreal_forced_aligner.corpus import AlignableCorpus, TranscribeCorpus
from montreal_forced_aligner.corpus.base import get_wav_info, SoxError
from montreal_forced_aligner.dictionary import Dictionary
from montreal_forced_aligner.config.train_config import train_yaml_to_config


def test_mp3(mp3_test_path):
    try:
        info = get_wav_info(mp3_test_path)
    except SoxError:
        pytest.skip()
    assert 'sox_string' in info


def test_add(basic_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, 'basic')
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    c = AlignableCorpus(basic_corpus_dir, output_directory, use_mp=True)
    assert 'test_add' not in c.utt_speak_mapping

    c.add_utterance('test_add', 'new_speaker', 'test_add', 'blah blah', 'wav_path')

    assert 'test_add' in c.utt_speak_mapping
    assert c.speak_utt_mapping['new_speaker'] == ['test_add']
    assert c.file_utt_mapping['test_add'] == ['test_add']
    assert c.text_mapping['test_add'] == 'blah blah'

    c.delete_utterance('test_add')
    assert 'test_add' not in c.utt_speak_mapping
    assert 'new_speaker' not in c.speak_utt_mapping
    assert 'test_add' not in c.file_utt_mapping
    assert 'test_add' not in c.text_mapping

def test_basic(basic_dict_path, basic_corpus_dir, generated_dir, default_feature_config):
    output_directory = os.path.join(generated_dir, 'basic')
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = Dictionary(basic_dict_path, output_directory)
    dictionary.write()
    c = AlignableCorpus(basic_corpus_dir, output_directory, use_mp=True)
    c.initialize_corpus(dictionary)
    default_feature_config.generate_features(c)
    assert c.get_feat_dim(default_feature_config) == 39
    dictionary.cleanup_logger()


def test_basic_txt(basic_corpus_txt_dir, basic_dict_path, generated_dir, default_feature_config):
    output_directory = os.path.join(generated_dir, 'basic')
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = Dictionary(basic_dict_path, os.path.join(generated_dir, 'basic'))
    dictionary.write()
    c = AlignableCorpus(basic_corpus_txt_dir, output_directory, use_mp=False)
    print(c.no_transcription_files)
    assert len(c.no_transcription_files) == 0
    c.initialize_corpus(dictionary)
    default_feature_config.generate_features(c)
    assert c.get_feat_dim(default_feature_config) == 39
    dictionary.cleanup_logger()


def test_alignable_from_temp(basic_corpus_txt_dir, basic_dict_path, generated_dir, default_feature_config):
    dictionary = Dictionary(basic_dict_path, os.path.join(generated_dir, 'basic'))
    dictionary.write()
    output_directory = os.path.join(generated_dir, 'basic')
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    c = AlignableCorpus(basic_corpus_txt_dir, output_directory, use_mp=False)
    assert len(c.no_transcription_files) == 0
    c.initialize_corpus(dictionary)
    default_feature_config.generate_features(c)
    assert c.get_feat_dim(default_feature_config) == 39

    c = AlignableCorpus(basic_corpus_txt_dir, output_directory, use_mp=False)
    assert len(c.no_transcription_files) == 0
    c.initialize_corpus(dictionary)
    default_feature_config.generate_features(c)
    assert c.get_feat_dim(default_feature_config) == 39
    dictionary.cleanup_logger()


def test_transcribe_from_temp(basic_corpus_txt_dir, basic_dict_path, generated_dir, default_feature_config):
    dictionary = Dictionary(basic_dict_path, os.path.join(generated_dir, 'basic'))
    dictionary.write()
    output_directory = os.path.join(generated_dir, 'basic')
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    c = TranscribeCorpus(basic_corpus_txt_dir, output_directory, use_mp=False)
    c.initialize_corpus(dictionary)
    default_feature_config.generate_features(c)
    assert c.get_feat_dim(default_feature_config) == 39

    c = TranscribeCorpus(basic_corpus_txt_dir, output_directory, use_mp=False)
    c.initialize_corpus(dictionary)
    default_feature_config.generate_features(c)
    assert c.get_feat_dim(default_feature_config) == 39
    dictionary.cleanup_logger()


def test_extra(sick_dict, extra_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, 'extra')
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    corpus = AlignableCorpus(extra_corpus_dir, output_directory, num_jobs=2, use_mp=False)
    corpus.initialize_corpus(sick_dict)


def test_stereo(basic_dict_path, stereo_corpus_dir, temp_dir, default_feature_config):
    temp = os.path.join(temp_dir, 'stereo')
    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = Dictionary(basic_dict_path, os.path.join(temp, 'basic'))
    dictionary.write()
    d = AlignableCorpus(stereo_corpus_dir, temp, use_mp=False)
    d.initialize_corpus(dictionary)
    default_feature_config.generate_features(d)
    assert d.get_feat_dim(default_feature_config) == 39
    dictionary.cleanup_logger()


def test_stereo_short_tg(basic_dict_path, stereo_corpus_short_tg_dir, temp_dir, default_feature_config):
    temp = os.path.join(temp_dir, 'stereo_tg')
    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = Dictionary(basic_dict_path, os.path.join(temp, 'basic'))
    dictionary.write()
    d = AlignableCorpus(stereo_corpus_short_tg_dir, temp, use_mp=False)
    d.initialize_corpus(dictionary)
    default_feature_config.generate_features(d)
    assert d.get_feat_dim(default_feature_config) == 39
    dictionary.cleanup_logger()


def test_flac(basic_dict_path, flac_corpus_dir, temp_dir, default_feature_config):
    temp = os.path.join(temp_dir, 'flac')
    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = Dictionary(basic_dict_path, os.path.join(temp, 'basic'))
    dictionary.write()
    d = AlignableCorpus(flac_corpus_dir, temp, use_mp=False)
    d.initialize_corpus(dictionary)
    default_feature_config.generate_features(d)
    assert d.get_feat_dim(default_feature_config) == 39
    dictionary.cleanup_logger()



def test_audio_directory(basic_dict_path, basic_split_dir, temp_dir, default_feature_config):
    temp = os.path.join(temp_dir, 'audio_dir_test')
    audio_dir, text_dir = basic_split_dir
    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = Dictionary(basic_dict_path, os.path.join(temp, 'basic'))
    dictionary.write()
    d = AlignableCorpus(text_dir, temp, use_mp=False, audio_directory=audio_dir)
    assert len(d.no_transcription_files) == 0
    assert len(d.utt_wav_mapping) > 0
    d.initialize_corpus(dictionary)
    default_feature_config.generate_features(d)
    assert d.get_feat_dim(default_feature_config) == 39

    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = Dictionary(basic_dict_path, os.path.join(temp, 'basic'))
    dictionary.write()
    d = AlignableCorpus(text_dir, temp, use_mp=True, audio_directory=audio_dir)
    assert len(d.no_transcription_files) == 0
    assert len(d.utt_wav_mapping) > 0
    d.initialize_corpus(dictionary)
    default_feature_config.generate_features(d)
    assert d.get_feat_dim(default_feature_config) == 39
    dictionary.cleanup_logger()


def test_flac_mp(basic_dict_path, flac_corpus_dir, temp_dir, default_feature_config):
    temp = os.path.join(temp_dir, 'flac')
    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = Dictionary(basic_dict_path, os.path.join(temp, 'basic'))
    dictionary.write()
    d = AlignableCorpus(flac_corpus_dir, temp, use_mp=True)
    d.initialize_corpus(dictionary)
    default_feature_config.generate_features(d)
    assert d.get_feat_dim(default_feature_config) == 39
    dictionary.cleanup_logger()


def test_flac_tg(basic_dict_path, flac_tg_corpus_dir, temp_dir, default_feature_config):
    temp = os.path.join(temp_dir, 'flac')
    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = Dictionary(basic_dict_path, os.path.join(temp, 'basic'))
    dictionary.write()
    d = AlignableCorpus(flac_tg_corpus_dir, temp, use_mp=False)
    d.initialize_corpus(dictionary)
    default_feature_config.generate_features(d)
    assert d.get_feat_dim(default_feature_config) == 39
    dictionary.cleanup_logger()


def test_flac_tg_mp(basic_dict_path, flac_tg_corpus_dir, temp_dir, default_feature_config):
    temp = os.path.join(temp_dir, 'flac')
    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = Dictionary(basic_dict_path, os.path.join(temp, 'basic'))
    dictionary.write()
    d = AlignableCorpus(flac_tg_corpus_dir, temp, use_mp=True)
    d.initialize_corpus(dictionary)
    default_feature_config.generate_features(d)
    assert d.get_feat_dim(default_feature_config) == 39
    dictionary.cleanup_logger()


def test_flac_tg_transcribe(basic_dict_path, flac_tg_corpus_dir, temp_dir, default_feature_config):
    temp = os.path.join(temp_dir, 'flac')
    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = Dictionary(basic_dict_path, os.path.join(temp, 'basic'))
    dictionary.write()
    d = TranscribeCorpus(flac_tg_corpus_dir, temp, use_mp=False)
    d.initialize_corpus(dictionary)
    default_feature_config.generate_features(d)
    assert d.get_feat_dim(default_feature_config) == 39
    dictionary.cleanup_logger()

    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = Dictionary(basic_dict_path, os.path.join(temp, 'basic'))
    dictionary.write()
    d = TranscribeCorpus(flac_tg_corpus_dir, temp, use_mp=True)
    d.initialize_corpus(dictionary)
    default_feature_config.generate_features(d)
    assert d.get_feat_dim(default_feature_config) == 39
    dictionary.cleanup_logger()


def test_flac_transcribe(basic_dict_path, flac_transcribe_corpus_dir, temp_dir, default_feature_config):
    temp = os.path.join(temp_dir, 'flac')
    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = Dictionary(basic_dict_path, os.path.join(temp, 'basic'))
    dictionary.write()
    d = TranscribeCorpus(flac_transcribe_corpus_dir, temp, use_mp=True)
    d.initialize_corpus(dictionary)
    default_feature_config.generate_features(d)
    assert d.get_feat_dim(default_feature_config) == 39
    dictionary.cleanup_logger()

    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = Dictionary(basic_dict_path, os.path.join(temp, 'basic'))
    dictionary.write()

    d = TranscribeCorpus(flac_transcribe_corpus_dir, temp, use_mp=False)
    d.initialize_corpus(dictionary)
    default_feature_config.generate_features(d)
    assert d.get_feat_dim(default_feature_config) == 39
    dictionary.cleanup_logger()


def test_24bit_wav(transcribe_corpus_24bit_dir, temp_dir, default_feature_config):
    temp = os.path.join(temp_dir, '24bit')
    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)

    c = TranscribeCorpus(transcribe_corpus_24bit_dir, temp, use_mp=False)
    assert len(c.unsupported_bit_depths) == 0
    c.initialize_corpus()
    default_feature_config.generate_features(c)
    assert c.get_feat_dim(default_feature_config) == 39
    assert len(c.utt_wav_mapping) == 2


def test_short_segments(basic_dict_path, shortsegments_corpus_dir, temp_dir, default_feature_config):
    temp = os.path.join(temp_dir, 'short_segments')
    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)
    dictionary = Dictionary(basic_dict_path, temp)
    dictionary.write()
    corpus = AlignableCorpus(shortsegments_corpus_dir, temp, use_mp=False)
    corpus.initialize_corpus(dictionary)
    default_feature_config.generate_features(corpus)
    assert len(corpus.feat_mapping.keys()) == 1
    assert len(corpus.utt_speak_mapping.keys()) == 3
    assert len(corpus.speak_utt_mapping.keys()) == 1
    assert len(corpus.text_mapping.keys()) == 3
    assert len(corpus.utt_wav_mapping.keys()) == 1
    assert len(corpus.segments.keys()) == 3
    print(corpus.segments)
    print(corpus.ignored_utterances)
    assert len(corpus.ignored_utterances) == 2
    dictionary.cleanup_logger()


def test_speaker_groupings(large_prosodylab_format_directory, temp_dir, large_dataset_dictionary, default_feature_config):
    output_directory = os.path.join(temp_dir, 'large')
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = Dictionary(large_dataset_dictionary, output_directory)
    dictionary.write()
    c = AlignableCorpus(large_prosodylab_format_directory, output_directory, use_mp=False)

    c.initialize_corpus(dictionary)
    default_feature_config.generate_features(c)
    speakers = os.listdir(large_prosodylab_format_directory)
    for s in speakers:
        assert any(s in x for x in c.speaker_groups)
    for root, dirs, files in os.walk(large_prosodylab_format_directory):
        for f in files:
            name, ext = os.path.splitext(f)
            assert any(name in x for x in c.groups)

    for root, dirs, files in os.walk(large_prosodylab_format_directory):
        for f in files:
            name, ext = os.path.splitext(f)
            assert any(name in x for x in c.feat_mapping)

    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary.write()
    c = AlignableCorpus(large_prosodylab_format_directory, output_directory, num_jobs=2, use_mp=False)

    c.initialize_corpus(dictionary)
    default_feature_config.generate_features(c)
    for s in speakers:
        assert any(s in x for x in c.speaker_groups)
    for root, dirs, files in os.walk(large_prosodylab_format_directory):
        for f in files:
            name, ext = os.path.splitext(f)
            assert any(name in x for x in c.groups)

    for root, dirs, files in os.walk(large_prosodylab_format_directory):
        for f in files:
            name, ext = os.path.splitext(f)
            assert any(name in x for x in c.feat_mapping)
    dictionary.cleanup_logger()


def test_subset(large_prosodylab_format_directory, temp_dir, large_dataset_dictionary, default_feature_config):
    output_directory = os.path.join(temp_dir, 'large_subset')
    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = Dictionary(large_dataset_dictionary, output_directory)
    dictionary.write()
    c = AlignableCorpus(large_prosodylab_format_directory, output_directory, use_mp=False)
    c.initialize_corpus(dictionary)
    sd = c.split_directory()

    default_feature_config.generate_features(c)
    s = c.subset_directory(10, default_feature_config)
    assert os.path.exists(sd)
    assert os.path.exists(s)
    dictionary.cleanup_logger()


def test_weird_words(weird_words_dir, temp_dir, sick_dict_path):
    output_directory = os.path.join(temp_dir, 'weird_words')
    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = Dictionary(sick_dict_path, output_directory)
    assert 'i’m' not in dictionary.words
    assert '’m' not in dictionary.words
    assert dictionary.words["i'm"][0]['pronunciation'] == ('ay', 'm', 'ih')
    assert dictionary.words["i'm"][1]['pronunciation'] == ('ay', 'm')
    assert dictionary.words["'m"][0]['pronunciation'] == ('m',)
    dictionary.write()
    c = AlignableCorpus(weird_words_dir, output_directory, use_mp=False)
    c.initialize_corpus(dictionary)
    print(c.utterance_oovs['weird-words'])
    assert c.utterance_oovs['weird-words'] == ['talking-ajfish', 'asds-asda', 'sdasd-me']

    dictionary.set_word_set(c.word_set)
    for w in ["i'm", "this'm", "sdsdsds'm", "'m"]:
        _ = dictionary.to_int(w)
    print(dictionary.oovs_found)
    assert "'m" not in dictionary.oovs_found
    dictionary.cleanup_logger()


def test_punctuated(punctuated_dir, temp_dir, sick_dict_path):
    output_directory = os.path.join(temp_dir, 'weird_words')
    shutil.rmtree(output_directory, ignore_errors=True)
    dictionary = Dictionary(sick_dict_path, output_directory)
    dictionary.write()
    c = AlignableCorpus(punctuated_dir, output_directory, use_mp=False)
    c.initialize_corpus(dictionary)
    print(c.text_mapping['punctuated'])
    assert c.text_mapping['punctuated'] == 'oh yes they they you know they love her and so i mean'
    dictionary.cleanup_logger()


def test_alternate_punctuation(punctuated_dir, temp_dir, sick_dict_path, different_punctuation_config):
    train_config, align_config = train_yaml_to_config(different_punctuation_config)
    output_directory = os.path.join(temp_dir, 'weird_words')
    shutil.rmtree(output_directory, ignore_errors=True)
    print(align_config.punctuation)
    dictionary = Dictionary(sick_dict_path, output_directory, punctuation=align_config.punctuation)
    dictionary.write()
    c = AlignableCorpus(punctuated_dir, output_directory, use_mp=False, punctuation=align_config.punctuation)
    print(c.punctuation)
    c.initialize_corpus(dictionary)
    print(c.text_mapping['punctuated'])
    assert c.text_mapping['punctuated'] == 'oh yes, they they, you know, they love her and so i mean'
    dictionary.cleanup_logger()

def test_xsampa_corpus(xsampa_corpus_dir, xsampa_dict_path, temp_dir, generated_dir, different_punctuation_config):
    train_config, align_config = train_yaml_to_config(different_punctuation_config)
    output_directory = os.path.join(temp_dir, 'xsampa_corpus')
    shutil.rmtree(output_directory, ignore_errors=True)
    print(align_config.punctuation)
    dictionary = Dictionary(xsampa_dict_path, output_directory, punctuation=align_config.punctuation)
    dictionary.write()
    c = AlignableCorpus(xsampa_corpus_dir, output_directory, use_mp=False, punctuation=align_config.punctuation)
    print(c.punctuation)
    c.initialize_corpus(dictionary)
    print(c.text_mapping['michael-xsampa'])
    assert c.text_mapping['michael-xsampa'] == r'@bUr\tOU {bstr\{kt {bSaIr\ Abr\utseIzi {br\@geItIN @bor\n {b3kr\Ambi {bI5s@`n Ar\g thr\Ip@5eI Ar\dvAr\k'.lower()
    dictionary.cleanup_logger()