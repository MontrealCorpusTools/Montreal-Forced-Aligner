import os
import pytest

from montreal_forced_aligner.command_line.transcribe import run_transcribe_corpus
from montreal_forced_aligner.command_line.mfa import parser


def test_transcribe(basic_corpus_dir, sick_dict_path, english_acoustic_model, generated_dir,
                    transcription_acoustic_model, transcription_language_model, temp_dir, transcribe_config):
    output_path = os.path.join(generated_dir, 'transcribe_test')
    command = ['transcribe', basic_corpus_dir, sick_dict_path, transcription_acoustic_model,
               transcription_language_model, output_path,
               '-t', temp_dir, '-q', '--clean', '-d', '--config_path', transcribe_config]
    args, unknown = parser.parse_known_args(command)
    run_transcribe_corpus(args)


def test_transcribe_speaker_dictionaries(multilingual_ipa_corpus_dir, ipa_speaker_dict_path, english_ipa_acoustic_model, generated_dir,
                    transcription_language_model, temp_dir, transcribe_config):
    output_path = os.path.join(generated_dir, 'transcribe_test')
    command = ['transcribe', multilingual_ipa_corpus_dir, ipa_speaker_dict_path, english_ipa_acoustic_model,
               transcription_language_model, output_path,
               '-t', temp_dir, '-q', '--clean', '-d', '--config_path', transcribe_config]
    args, unknown = parser.parse_known_args(command)
    run_transcribe_corpus(args)


def test_transcribe_speaker_dictionaries_evaluate(multilingual_ipa_corpus_dir, ipa_speaker_dict_path, english_ipa_acoustic_model, generated_dir,
                    transcription_language_model, temp_dir, transcribe_config):
    output_path = os.path.join(generated_dir, 'transcribe_test')
    command = ['transcribe', multilingual_ipa_corpus_dir, ipa_speaker_dict_path, english_ipa_acoustic_model,
               transcription_language_model, output_path,
               '-t', temp_dir, '-q', '--clean', '-d', '--config_path', transcribe_config, '--evaluate']
    args, unknown = parser.parse_known_args(command)
    run_transcribe_corpus(args)