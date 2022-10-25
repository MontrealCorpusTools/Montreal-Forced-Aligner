import os

import click.testing

from montreal_forced_aligner.command_line.mfa import mfa_cli


def test_create_segments(
    basic_corpus_dir,
    generated_dir,
    temp_dir,
    basic_segment_config_path,
):
    output_path = os.path.join(generated_dir, "segment_output")
    command = [
        "segment",
        basic_corpus_dir,
        output_path,
        "-t",
        os.path.join(temp_dir, "sad_cli"),
        "-q",
        "--clean",
        "--debug",
        "-v",
        "--config_path",
        basic_segment_config_path,
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
    assert os.path.exists(os.path.join(output_path, "michael", "acoustic_corpus.TextGrid"))
