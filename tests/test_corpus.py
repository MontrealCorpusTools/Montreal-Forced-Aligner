import os
import pytest
import shutil

from aligner.corpus import Corpus
from aligner.config import MfccConfig
from aligner.dictionary import Dictionary


def test_basic(basic_dict_path, basic_dir, generated_dir):
    dictionary = Dictionary(basic_dict_path, os.path.join(generated_dir, 'basic'))
    dictionary.write()
    output_directory = os.path.join(generated_dir, 'basic')
    d = Corpus(basic_dir, output_directory)
    d.write()
    d.create_mfccs()
    d.setup_splits(dictionary)
    assert(d.get_feat_dim() == '39')


def test_extra(sick_dict, extra_dir, generated_dir):
    output_directory = os.path.join(generated_dir, 'extra')
    corpus = Corpus(extra_dir, output_directory, num_jobs = 2)
    corpus.write()
    corpus.create_mfccs()
    corpus.setup_splits(sick_dict)



def test_stereo(basic_dict_path, textgrid_directory, generated_dir):
    temp = os.path.join(generated_dir, 'stereo')
    dictionary = Dictionary(basic_dict_path, os.path.join(temp, 'basic'))
    dictionary.write()
    d = Corpus(os.path.join(textgrid_directory, 'stereo'), temp)
    d.write()
    d.create_mfccs()
    d.setup_splits(dictionary)
    assert(d.get_feat_dim() == '39')


def test_short_segments(textgrid_directory, generated_dir):
    temp = os.path.join(generated_dir, 'short_segments')
    corpus = Corpus(os.path.join(textgrid_directory, 'short_segments'), temp)
    corpus.write()
    corpus.create_mfccs()
    assert(len(corpus.feat_mapping.keys()) == 2)
    assert(len(corpus.utt_speak_mapping.keys()) == 2)
    assert(len(corpus.speak_utt_mapping.keys()) == 1)
    assert(len(corpus.text_mapping.keys()) == 2)
    assert(len(corpus.utt_wav_mapping.keys()) == 1)
    assert(len(corpus.segments.keys()) == 2)
    assert(len(corpus.ignored_utterances) == 1)


def test_speaker_groupings(large_prosodylab_format_directory, generated_dir):
    output_directory = os.path.join(generated_dir, 'large')
    shutil.rmtree(output_directory, ignore_errors = True)
    c = Corpus(large_prosodylab_format_directory, output_directory)
    speakers = os.listdir(large_prosodylab_format_directory)
    for s in speakers:
        assert(any(s in x for x in c.speaker_groups))
    for root, dirs, files in os.walk(large_prosodylab_format_directory):
        for f in files:
            name, ext = os.path.splitext(f)
            assert(any(name in x for x in c.groups))
    c.create_mfccs()
    for root, dirs, files in os.walk(large_prosodylab_format_directory):
        for f in files:
            name, ext = os.path.splitext(f)
            assert(any(name in x for x in c.feat_mapping))

    shutil.rmtree(output_directory, ignore_errors = True)
    c = Corpus(large_prosodylab_format_directory, output_directory, num_jobs = 2)
    for s in speakers:
        assert(any(s in x for x in c.speaker_groups))
    for root, dirs, files in os.walk(large_prosodylab_format_directory):
        for f in files:
            name, ext = os.path.splitext(f)
            assert(any(name in x for x in c.groups))
    c.create_mfccs()
    for root, dirs, files in os.walk(large_prosodylab_format_directory):
        for f in files:
            name, ext = os.path.splitext(f)
            assert(any(name in x for x in c.feat_mapping))
