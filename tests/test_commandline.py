import os
import pytest

from aligner.command_line.align import align_corpus, align_included_model

def test_align_large_prosodylab(large_prosodylab_format_directory, prosodylab_output_directory):
    language = 'english'
    corpus_dir = large_prosodylab_format_directory
    output_directory = prosodylab_output_directory
    speaker_characters = 0
    num_jobs = 0
    verbose = False
    align_included_model(language, corpus_dir,  output_directory,
                            speaker_characters, num_jobs, verbose)
    for root, dirs, files in os.walk(large_prosodylab_format_directory):
        new_root = root.replace(large_prosodylab_format_directory, prosodylab_output_directory)
        for d in dirs:
            assert(os.path.exists(os.path.join(new_root, d)))
        for f in files:
            if not f.endswith('.wav'):
                continue
            new_f = f.replace('.wav', '.TextGrid')
            assert(os.path.exists(os.path.join(new_root, new_f)))

