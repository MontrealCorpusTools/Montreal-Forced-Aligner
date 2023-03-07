import os
import shutil

import click.testing
import pytest

from montreal_forced_aligner.command_line.mfa import mfa_cli
from montreal_forced_aligner.diarization.speaker_diarizer import FOUND_SPEECHBRAIN


def test_create_segments(
    basic_corpus_dir,
    generated_dir,
    temp_dir,
    basic_segment_config_path,
    db_setup,
):
    output_path = generated_dir.joinpath("segment_output")
    shutil.rmtree(output_path, ignore_errors=True)
    command = [
        "segment",
        basic_corpus_dir,
        output_path,
        "-q",
        "--clean",
        "--debug",
        "-v",
        "--config_path",
        basic_segment_config_path,
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
    assert os.path.exists(os.path.join(output_path, "michael", "acoustic_corpus.TextGrid"))


def test_create_segments_speechbrain(
    basic_corpus_dir,
    generated_dir,
    temp_dir,
    basic_segment_config_path,
    db_setup,
):
    if not FOUND_SPEECHBRAIN:
        pytest.skip("SpeechBrain not installed")
    output_path = generated_dir.joinpath("segment_output")
    command = [
        "segment",
        basic_corpus_dir,
        output_path,
        "-q",
        "--clean",
        "--no_debug",
        "-v",
        "--speechbrain",
        "--config_path",
        basic_segment_config_path,
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
    assert os.path.exists(os.path.join(output_path, "michael", "acoustic_corpus.TextGrid"))
