import os
import sys
import pytest
import shutil

from montreal_forced_aligner.corpus import AlignableCorpus, TranscribeCorpus
from montreal_forced_aligner.dictionary import Dictionary
from montreal_forced_aligner.exceptions import CorpusError


def test_basic(basic_dict_path, basic_corpus_dir, generated_dir, default_feature_config):
    dictionary = Dictionary(basic_dict_path, os.path.join(generated_dir, 'basic'))
    dictionary.write()
    output_directory = os.path.join(generated_dir, 'basic')
    c = AlignableCorpus(basic_corpus_dir, output_directory)
    c.initialize_corpus(dictionary)
    default_feature_config.generate_features(c)
    assert c.get_feat_dim(default_feature_config) == 39


def test_basic_txt(basic_corpus_txt_dir, basic_dict_path, generated_dir, default_feature_config):
    dictionary = Dictionary(basic_dict_path, os.path.join(generated_dir, 'basic'))
    dictionary.write()
    output_directory = os.path.join(generated_dir, 'basic')
    c = AlignableCorpus(basic_corpus_txt_dir, output_directory)
    assert len(c.no_transcription_files) == 0
    c.initialize_corpus(dictionary)
    default_feature_config.generate_features(c)
    assert c.get_feat_dim(default_feature_config) == 39


def test_extra(sick_dict, extra_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, 'extra')
    corpus = AlignableCorpus(extra_corpus_dir, output_directory, num_jobs=2)
    corpus.initialize_corpus(sick_dict)


def test_stereo(basic_dict_path, stereo_corpus_dir, temp_dir, default_feature_config):
    temp = os.path.join(temp_dir, 'stereo')
    dictionary = Dictionary(basic_dict_path, os.path.join(temp, 'basic'))
    dictionary.write()
    d = AlignableCorpus(stereo_corpus_dir, temp)
    d.initialize_corpus(dictionary)
    default_feature_config.generate_features(d)
    assert d.get_feat_dim(default_feature_config) == 39


def test_24bit_wav(transcribe_corpus_24bit_dir, temp_dir, default_feature_config):
    temp = os.path.join(temp_dir, '24bit')

    c = TranscribeCorpus(transcribe_corpus_24bit_dir, temp)
    assert len(c.unsupported_bit_depths) == 1
    with pytest.raises(CorpusError):
        c.initialize_corpus()


def test_short_segments(basic_dict_path, shortsegments_corpus_dir, temp_dir, default_feature_config):
    temp = os.path.join(temp_dir, 'short_segments')
    dictionary = Dictionary(basic_dict_path, temp)
    dictionary.write()
    corpus = AlignableCorpus(shortsegments_corpus_dir, temp)
    corpus.initialize_corpus(dictionary)
    default_feature_config.generate_features(corpus)
    assert len(corpus.feat_mapping.keys()) == 2
    assert len(corpus.utt_speak_mapping.keys()) == 3
    assert len(corpus.speak_utt_mapping.keys()) == 1
    assert len(corpus.text_mapping.keys()) == 3
    assert len(corpus.utt_wav_mapping.keys()) == 1
    assert len(corpus.segments.keys()) == 3
    assert len(corpus.ignored_utterances) == 1


def test_speaker_groupings(large_prosodylab_format_directory, temp_dir, large_dataset_dictionary, default_feature_config):
    output_directory = os.path.join(temp_dir, 'large')
    shutil.rmtree(output_directory, ignore_errors=True)
    d = Dictionary(large_dataset_dictionary, output_directory)
    d.write()
    c = AlignableCorpus(large_prosodylab_format_directory, output_directory)

    c.initialize_corpus(d)
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
    d.write()
    c = AlignableCorpus(large_prosodylab_format_directory, output_directory, num_jobs=2)

    c.initialize_corpus(d)
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


def test_subset(large_prosodylab_format_directory, temp_dir, large_dataset_dictionary, default_feature_config):
    output_directory = os.path.join(temp_dir, 'large_subset')
    shutil.rmtree(output_directory, ignore_errors=True)
    d = Dictionary(large_dataset_dictionary, output_directory)
    d.write()
    c = AlignableCorpus(large_prosodylab_format_directory, output_directory)
    c.initialize_corpus(d)
    sd = c.split_directory()

    default_feature_config.generate_features(c)
    s = c.subset_directory(10, default_feature_config)
    assert os.path.exists(sd)
    assert os.path.exists(s)
