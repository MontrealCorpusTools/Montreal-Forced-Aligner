import os

import click.testing
import sqlalchemy.orm

from montreal_forced_aligner.command_line.mfa import mfa_cli


def test_train_dict(
    basic_corpus_dir,
    english_dictionary,
    english_acoustic_model,
    generated_dir,
    temp_dir,
    basic_align_config_path,
    db_setup,
):
    output_path = generated_dir.joinpath("trained_dict")
    command = [
        "train_dictionary",
        basic_corpus_dir,
        english_dictionary,
        english_acoustic_model,
        output_path,
        "-q",
        "--clean",
        "--debug",
        "--silence_probabilities",
        "--config_path",
        basic_align_config_path,
        "--use_mp",
    ]
    command = [str(x) for x in command]
    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value

    dict_path = os.path.join(output_path, "english_us_arpa.dict")
    assert os.path.exists(output_path)
    sqlalchemy.orm.close_all_sessions()
    textgrid_output = generated_dir.joinpath("trained_dict_output")
    command = [
        "align",
        basic_corpus_dir,
        dict_path,
        english_acoustic_model,
        textgrid_output,
        "-q",
        "--clean",
        "--debug",
        "--config_path",
        basic_align_config_path,
    ]
    command = [str(x) for x in command]
    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value
    assert os.path.exists(textgrid_output)
