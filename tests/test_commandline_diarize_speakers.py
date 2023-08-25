import os

import click.testing
import pytest

from montreal_forced_aligner.command_line.mfa import mfa_cli
from montreal_forced_aligner.diarization.speaker_diarizer import FOUND_SPEECHBRAIN
from montreal_forced_aligner.exceptions import DatabaseError


def test_cluster_mfa_no_postgres(
    combined_corpus_dir,
    multilingual_ivector_model,
    generated_dir,
    transcription_acoustic_model,
    transcription_language_model,
    temp_dir,
    db_setup,
):
    output_path = generated_dir.joinpath("cluster_test_mfa")
    command = [
        "diarize",
        combined_corpus_dir,
        multilingual_ivector_model,
        output_path,
        "--cluster",
        "--cluster_type",
        "kmeans",
        "--use_postgres",
        "--expected_num_speakers",
        "3",
        "--clean",
        "--evaluate",
        "--no_use_postgres",
    ]
    command = [str(x) for x in command]
    with pytest.raises(DatabaseError):
        result = click.testing.CliRunner(mix_stderr=False).invoke(
            mfa_cli, command, catch_exceptions=True
        )
        print(result.stdout)
        print(result.stderr)
        if result.exception:
            print(result.exc_info)
            raise result.exception


def test_cluster_mfa(
    combined_corpus_dir,
    multilingual_ivector_model,
    generated_dir,
    transcription_acoustic_model,
    transcription_language_model,
    temp_dir,
    db_setup,
):
    output_path = generated_dir.joinpath("cluster_test_mfa")
    command = [
        "diarize",
        combined_corpus_dir,
        multilingual_ivector_model,
        output_path,
        "--cluster",
        "--cluster_type",
        "kmeans",
        "--use_postgres",
        "--expected_num_speakers",
        "3",
        "--clean",
        "--evaluate",
    ]
    command = [str(x) for x in command]
    result = click.testing.CliRunner(mix_stderr=False).invoke(
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
    db_setup,
):
    output_path = generated_dir.joinpath("classify_test_mfa")
    command = [
        "diarize",
        combined_corpus_dir,
        multilingual_ivector_model,
        output_path,
        "--classify",
        "--clean",
        "--evaluate",
        "--use_postgres",
    ]
    command = [str(x) for x in command]
    result = click.testing.CliRunner(mix_stderr=False).invoke(
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
    db_setup,
):
    if not FOUND_SPEECHBRAIN:
        pytest.skip("SpeechBrain not installed")
    output_path = generated_dir.joinpath("cluster_test_sb")
    command = [
        "diarize",
        combined_corpus_dir,
        "speechbrain",
        output_path,
        "--cluster",
        "--cluster_type",
        "kmeans",
        "--expected_num_speakers",
        "3",
        "--clean",
        "--no_use_pca",
        "--no_debug",
        "--evaluate",
        "--use_postgres",
    ]
    command = [str(x) for x in command]
    result = click.testing.CliRunner(mix_stderr=False).invoke(
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
    db_setup,
):
    if not FOUND_SPEECHBRAIN:
        pytest.skip("SpeechBrain not installed")
    output_path = generated_dir.joinpath("classify_test_sb")
    command = [
        "diarize",
        combined_corpus_dir,
        "speechbrain",
        output_path,
        "--classify",
        "--clean",
        "--no_debug",
        "--evaluate",
        "--use_postgres",
    ]
    command = [str(x) for x in command]
    result = click.testing.CliRunner(mix_stderr=False).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value
    assert os.path.exists(output_path)
