import os

from montreal_forced_aligner.command_line.mfa import parser
from montreal_forced_aligner.command_line.train_dictionary import run_train_dictionary


def test_train_dict(
    basic_corpus_dir,
    sick_dict_path,
    english_acoustic_model,
    generated_dir,
    transcription_acoustic_model,
    transcription_language_model,
    temp_dir,
    basic_align_config_path,
):
    output_path = os.path.join(generated_dir, "trained_dict")
    command = [
        "train_dictionary",
        basic_corpus_dir,
        sick_dict_path,
        transcription_acoustic_model,
        output_path,
        "-t",
        temp_dir,
        "-q",
        "--clean",
        "--debug",
        "--config_path",
        basic_align_config_path,
    ]
    args, unknown = parser.parse_known_args(command)
    run_train_dictionary(args)
