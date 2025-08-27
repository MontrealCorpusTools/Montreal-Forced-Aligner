import os

import click.testing

from montreal_forced_aligner.command_line.mfa import mfa_cli


def test_remap_dictionary(
    english_us_mfa_dictionary,
    english_arpa_remapping_path,
    english_acoustic_model,
    generated_dir,
    temp_dir,
    db_setup,
):
    output_path = generated_dir.joinpath("remapped_dictionary.txt")
    command = [
        "remap_dictionary",
        english_us_mfa_dictionary,
        english_acoustic_model,
        english_arpa_remapping_path,
        output_path,
        "-q",
        "--clean",
        "--debug",
        "--no_use_mp",
        "--no_use_postgres",
        "-v",
    ]
    command = [str(x) for x in command]
    result = click.testing.CliRunner().invoke(mfa_cli, command, catch_exceptions=True)
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value

    assert os.path.exists(output_path)


def test_remap_alignments(
    mfa_example_aligned_dir,
    english_arpa_remapping_path,
    generated_dir,
    temp_dir,
    db_setup,
):
    output_path = generated_dir.joinpath("remapped_alignments")
    command = [
        "remap",
        "alignments",
        mfa_example_aligned_dir,
        english_arpa_remapping_path,
        output_path,
        "-q",
        "--clean",
        "--debug",
        "--no_use_mp",
        "--no_use_postgres",
        "-v",
    ]
    command = [str(x) for x in command]
    result = click.testing.CliRunner().invoke(mfa_cli, command, catch_exceptions=True)
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value

    assert os.path.exists(os.path.join(output_path, "michael", "mfa_michael.TextGrid"))
