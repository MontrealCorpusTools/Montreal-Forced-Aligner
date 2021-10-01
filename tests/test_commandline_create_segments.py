import os
import pytest

from montreal_forced_aligner.command_line.create_segments import run_create_segments
from montreal_forced_aligner.command_line.mfa import parser


def test_create_segments(basic_corpus_dir, sick_dict_path, english_acoustic_model, generated_dir,
                    transcription_acoustic_model, transcription_language_model, temp_dir, basic_segment_config):
    output_path = os.path.join(generated_dir, 'segment_output')
    command = ['create_segments', basic_corpus_dir,
               output_path,
               '-t', temp_dir, '-q', '--clean', '--debug', '-v', '--config_path', basic_segment_config]
    args, unknown = parser.parse_known_args(command)
    run_create_segments(args)