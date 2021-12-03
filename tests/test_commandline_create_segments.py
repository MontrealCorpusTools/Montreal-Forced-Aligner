import os

from montreal_forced_aligner.command_line.create_segments import run_create_segments
from montreal_forced_aligner.command_line.mfa import parser


def test_create_segments(
    basic_corpus_dir,
    generated_dir,
    temp_dir,
    basic_segment_config_path,
):
    output_path = os.path.join(generated_dir, "segment_output")
    command = [
        "create_segments",
        basic_corpus_dir,
        output_path,
        "-t",
        temp_dir,
        "-q",
        "--clean",
        "--debug",
        "-v",
        "--config_path",
        basic_segment_config_path,
    ]
    args, unknown = parser.parse_known_args(command)
    run_create_segments(args)
    assert os.path.exists(os.path.join(output_path, "michael", "acoustic_corpus.TextGrid"))
