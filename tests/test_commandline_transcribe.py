import os
import pytest

from montreal_forced_aligner.command_line.transcribe import run_transcribe_corpus


class DummyArgs(object):
    def __init__(self):
        self.corpus_directory = ''
        self.dictionary_path = ''
        self.acoustic_model_path = ''
        self.output_directory = ''
        self.config_path = ''
        self.speaker_characters = 0
        self.num_jobs = 0
        self.verbose = False
        self.clean = True
        self.fast = True
        self.debug = False
        self.evaluate = False
        self.temp_directory = None


def test_transcribe(basic_corpus_dir, sick_dict_path, english_acoustic_model, generated_dir,
                    transcription_acoustic_model, transcription_language_model):
    output_path = os.path.join(generated_dir, 'transcribe_test')
    args = DummyArgs()
    args.acoustic_model_path = transcription_acoustic_model
    args.corpus_directory = basic_corpus_dir
    args.dictionary_path = sick_dict_path
    args.language_model_path = transcription_language_model
    args.output_directory = output_path
    args.evaluate = True
    run_transcribe_corpus(args)
