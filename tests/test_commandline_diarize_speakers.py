import os

import click.testing
import pytest

from montreal_forced_aligner.command_line.mfa import mfa_cli
from montreal_forced_aligner.diarization.speaker_diarizer import FOUND_SPEECHBRAIN


def test_cluster_mfa(
    combined_corpus_dir,
    multilingual_ivector_model,
    generated_dir,
    transcription_acoustic_model,
    transcription_language_model,
    temp_dir,
):
    output_path = os.path.join(generated_dir, "cluster_test_mfa")
    command = [
        "diarize",
        combined_corpus_dir,
        multilingual_ivector_model,
        output_path,
        "-t",
        os.path.join(temp_dir, "diarize_cli"),
        "--cluster",
        "--cluster_type",
        "kmeans",
        "--expected_num_speakers",
        "3",
        "--clean",
        "--evaluate",
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
    assert os.path.exists(output_path)


def test_classify_mfa(
    combined_corpus_dir,
    multilingual_ivector_model,
    generated_dir,
    transcription_acoustic_model,
    transcription_language_model,
    temp_dir,
):
    output_path = os.path.join(generated_dir, "classify_test_mfa")
    command = [
        "diarize",
        combined_corpus_dir,
        multilingual_ivector_model,
        output_path,
        "-t",
        os.path.join(temp_dir, "diarize_cli"),
        "--classify",
        "--clean",
        "--evaluate",
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
    assert os.path.exists(output_path)


def test_cluster_speechbrain(
    combined_corpus_dir,
    generated_dir,
    transcription_acoustic_model,
    transcription_language_model,
    temp_dir,
):
    if not FOUND_SPEECHBRAIN:
        pytest.skip("SpeechBrain not installed")
    output_path = os.path.join(generated_dir, "cluster_test_sb")
    command = [
        "diarize",
        combined_corpus_dir,
        "speechbrain",
        output_path,
        "-t",
        os.path.join(temp_dir, "diarize_cli"),
        "--cluster",
        "--cluster_type",
        "kmeans",
        "--expected_num_speakers",
        "3",
        "--clean",
        "--no_use_pca",
        "--no_debug",
        "--evaluate",
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
    assert os.path.exists(output_path)


def test_classify_speechbrain(
    combined_corpus_dir,
    generated_dir,
    transcription_acoustic_model,
    transcription_language_model,
    temp_dir,
):
    if not FOUND_SPEECHBRAIN:
        pytest.skip("SpeechBrain not installed")
    output_path = os.path.join(generated_dir, "classify_test_sb")
    command = [
        "diarize",
        combined_corpus_dir,
        "speechbrain",
        output_path,
        "-t",
        os.path.join(temp_dir, "diarize_cli"),
        "--classify",
        "--clean",
        "--no_debug",
        "--evaluate",
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
    assert os.path.exists(output_path)
