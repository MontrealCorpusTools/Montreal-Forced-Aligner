import os

import click.testing

from montreal_forced_aligner.command_line.mfa import mfa_cli


def test_cluster_mfa(
    basic_corpus_dir,
    english_ivector_model,
    generated_dir,
    transcription_acoustic_model,
    transcription_language_model,
    temp_dir,
):
    output_path = os.path.join(generated_dir, "cluster_test_mfa")
    command = [
        "diarize",
        basic_corpus_dir,
        english_ivector_model,
        output_path,
        "-t",
        os.path.join(temp_dir, "diarize_cli"),
        "--cluster",
        "--cluster_type",
        "affinity",
        "--clean",
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
    basic_corpus_dir,
    english_ivector_model,
    generated_dir,
    transcription_acoustic_model,
    transcription_language_model,
    temp_dir,
):
    output_path = os.path.join(generated_dir, "classify_test_mfa")
    command = [
        "diarize",
        basic_corpus_dir,
        english_ivector_model,
        output_path,
        "-t",
        os.path.join(temp_dir, "diarize_cli"),
        "--classify",
        "--clean",
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
    basic_corpus_dir,
    generated_dir,
    transcription_acoustic_model,
    transcription_language_model,
    temp_dir,
):
    output_path = os.path.join(generated_dir, "cluster_test_sb")
    command = [
        "diarize",
        basic_corpus_dir,
        "speechbrain",
        output_path,
        "-t",
        os.path.join(temp_dir, "diarize_cli"),
        "--cluster",
        "--cluster_type",
        "affinity",
        "--clean",
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
    basic_corpus_dir,
    generated_dir,
    transcription_acoustic_model,
    transcription_language_model,
    temp_dir,
):
    output_path = os.path.join(generated_dir, "classify_test_sb")
    command = [
        "diarize",
        basic_corpus_dir,
        "speechbrain",
        output_path,
        "-t",
        os.path.join(temp_dir, "diarize_cli"),
        "--classify",
        "--clean",
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
