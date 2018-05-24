import os
import sys
import pytest
import shutil

sys.path.insert(0,"/Users/mlml/Documents/GitHub/Montreal-Forced-Aligner")
from aligner.corpus import Corpus
from aligner.config import MfccConfig
from aligner.dictionary import Dictionary

def test_basic(basic_dict_path, basic_corpus_dir, generated_dir):
    dictionary = Dictionary(basic_dict_path, os.path.join(generated_dir, 'basic'))
    dictionary.write()
    output_directory = os.path.join(generated_dir, 'basic')
    d = Corpus(basic_corpus_dir, output_directory)
    d.initialize_corpus(dictionary)
    #assert(1==2)    # Dumb assert to make print
    assert (d.get_feat_dim() == '39')


def test_extra(sick_dict, extra_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, 'extra')
    corpus = Corpus(extra_corpus_dir, output_directory, num_jobs=2)
    corpus.initialize_corpus(sick_dict)


def test_stereo(basic_dict_path, stereo_corpus_dir, temp_dir):
    temp = os.path.join(temp_dir, 'stereo')
    dictionary = Dictionary(basic_dict_path, os.path.join(temp, 'basic'))
    dictionary.write()
    d = Corpus(stereo_corpus_dir, temp)
    d.initialize_corpus(dictionary)
    assert (d.get_feat_dim() == '39')


def test_short_segments(basic_dict_path, shortsegments_corpus_dir, temp_dir):
    temp = os.path.join(temp_dir, 'short_segments')
    dictionary = Dictionary(basic_dict_path, temp)
    dictionary.write()
    corpus = Corpus(shortsegments_corpus_dir, temp)
    corpus.initialize_corpus(dictionary)
    assert (len(corpus.feat_mapping.keys()) == 2)
    assert (len(corpus.utt_speak_mapping.keys()) == 2)
    assert (len(corpus.speak_utt_mapping.keys()) == 1)
    assert (len(corpus.text_mapping.keys()) == 2)
    assert (len(corpus.utt_wav_mapping.keys()) == 1)
    assert (len(corpus.segments.keys()) == 2)
    assert (len(corpus.ignored_utterances) == 1)


def test_speaker_groupings(large_prosodylab_format_directory, temp_dir):
    output_directory = os.path.join(temp_dir, 'large')
    shutil.rmtree(output_directory, ignore_errors=True)
    c = Corpus(large_prosodylab_format_directory, output_directory)
    speakers = os.listdir(large_prosodylab_format_directory)
    for s in speakers:
        assert (any(s in x for x in c.speaker_groups))
    for root, dirs, files in os.walk(large_prosodylab_format_directory):
        for f in files:
            name, ext = os.path.splitext(f)
            assert (any(name in x for x in c.groups))
    c.create_mfccs()
    for root, dirs, files in os.walk(large_prosodylab_format_directory):
        for f in files:
            name, ext = os.path.splitext(f)
            assert (any(name in x for x in c.feat_mapping))

    shutil.rmtree(output_directory, ignore_errors=True)
    c = Corpus(large_prosodylab_format_directory, output_directory, num_jobs=2)
    for s in speakers:
        assert (any(s in x for x in c.speaker_groups))
    for root, dirs, files in os.walk(large_prosodylab_format_directory):
        for f in files:
            name, ext = os.path.splitext(f)
            assert (any(name in x for x in c.groups))
    c.create_mfccs()
    for root, dirs, files in os.walk(large_prosodylab_format_directory):
        for f in files:
            name, ext = os.path.splitext(f)
            assert (any(name in x for x in c.feat_mapping))
