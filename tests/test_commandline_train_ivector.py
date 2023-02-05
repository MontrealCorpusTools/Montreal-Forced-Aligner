import os

import click.testing

from montreal_forced_aligner.command_line.mfa import mfa_cli


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
        os.path.join(temp_dir, "train_ivector_cli"),
        "--config_path",
        train_ivector_config_path,
        "-q",
        "--clean",
        "--debug",
    ]
    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value
    assert os.path.exists(ivector_output_model_path)
