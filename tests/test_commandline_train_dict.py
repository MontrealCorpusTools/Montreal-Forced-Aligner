import os

from montreal_forced_aligner.command_line.align import run_align_corpus
from montreal_forced_aligner.command_line.mfa import parser
from montreal_forced_aligner.command_line.train_dictionary import run_train_dictionary


def test_train_dict(
    basic_corpus_dir,
    english_dictionary,
    english_acoustic_model,
    generated_dir,
    temp_dir,
    basic_align_config_path,
):
    output_path = os.path.join(generated_dir, "trained_dict")
    command = [
        "train_dictionary",
        basic_corpus_dir,
        english_dictionary,
        english_acoustic_model,
        output_path,
        "-t",
        temp_dir,
        "-q",
        "--clean",
        "--debug",
        "--silence_probabilities",
        "--config_path",
        basic_align_config_path,
        "--use_mp",
    ]
    args, unknown = parser.parse_known_args(command)
    run_train_dictionary(args)

    dict_path = os.path.join(output_path, "english_us_arpa.dict")
    assert os.path.exists(output_path)
    textgrid_output = os.path.join(generated_dir, "trained_dict_output")
    command = [
        "align",
        basic_corpus_dir,
        dict_path,
        english_acoustic_model,
        textgrid_output,
        "-t",
        temp_dir,
        "-q",
        "--clean",
        "--debug",
        "--config_path",
        basic_align_config_path,
    ]
    args, unknown = parser.parse_known_args(command)
    run_align_corpus(args)
