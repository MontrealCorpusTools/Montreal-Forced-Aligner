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
