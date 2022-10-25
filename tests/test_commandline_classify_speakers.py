import os

import click.testing
import pytest

from montreal_forced_aligner.command_line.mfa import mfa_cli


@pytest.mark.skip("Speaker diarization functionality disabled.")
def test_cluster(
    basic_corpus_dir,
    english_ivector_model,
    generated_dir,
    transcription_acoustic_model,
    transcription_language_model,
    temp_dir,
):
    output_path = os.path.join(generated_dir, "cluster_test")
    command = [
        "classify_speakers",
        basic_corpus_dir,
        "english_ivector",
        output_path,
        "-t",
        os.path.join(temp_dir, "diarize_cli"),
        "-q",
        "--clean",
        "--debug",
        "--cluster",
        "-s",
        "2",
        "--disable_mp",
    ]
    click.testing.CliRunner().invoke(mfa_cli, command, catch_exceptions=False)
    assert os.path.exists(output_path)
