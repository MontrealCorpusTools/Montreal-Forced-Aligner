import os

from montreal_forced_aligner.command_line.mfa import parser
from montreal_forced_aligner.command_line.train_ivector_extractor import (
    run_train_ivector_extractor,
)


def test_basic_ivector(
    basic_corpus_dir,
    generated_dir,
    temp_dir,
    train_ivector_config_path,
    ivector_output_model_path,
):
    command = [
        "train_ivector",
        basic_corpus_dir,
        ivector_output_model_path,
        "-t",
        temp_dir,
        "--config_path",
        train_ivector_config_path,
        "-q",
        "--clean",
        "--debug",
    ]
    args, unknown = parser.parse_known_args(command)
    run_train_ivector_extractor(args)
    assert os.path.exists(args.output_model_path)
