import os

import pytest

from montreal_forced_aligner.command_line.classify_speakers import run_classify_speakers
from montreal_forced_aligner.command_line.mfa import parser


def test_cluster(
    basic_corpus_dir,
    sick_dict_path,
    english_ivector_model,
    generated_dir,
    transcription_acoustic_model,
    transcription_language_model,
    temp_dir,
):
    pytest.skip("Speaker diarization functionality disabled.")
    output_path = os.path.join(generated_dir, "cluster_test")
    command = [
        "classify_speakers",
        basic_corpus_dir,
        "english_ivector",
        output_path,
        "-t",
        temp_dir,
        "-q",
        "--clean",
        "--debug",
        "--cluster",
        "-s",
        "2",
        "--disable_mp",
    ]
    args, unknown = parser.parse_known_args(command)
    run_classify_speakers(args)
