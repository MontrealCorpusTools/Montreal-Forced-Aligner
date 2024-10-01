import os

import click.testing
import pytest

from montreal_forced_aligner.command_line.mfa import mfa_cli
from montreal_forced_aligner.exceptions import PhoneGroupTopologyMismatchError


@pytest.mark.skip("Inconsistent failing on CI")
def test_train_acoustic_with_g2p(
    combined_corpus_dir,
    english_us_mfa_reduced_dict,
    generated_dir,
    temp_dir,
    train_g2p_acoustic_config_path,
    acoustic_g2p_model_path,
    db_setup,
):
    if os.path.exists(acoustic_g2p_model_path):
        os.remove(acoustic_g2p_model_path)
    output_directory = generated_dir.joinpath("train_g2p_textgrids")
    command = [
        "train",
        combined_corpus_dir,
        english_us_mfa_reduced_dict,
        acoustic_g2p_model_path,
        "--output_directory",
        output_directory,
        "-q",
        "--clean",
        "--quiet",
        "--debug",
        "--use_postgres",
        "--no_use_mp",
        "--config_path",
        train_g2p_acoustic_config_path,
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
    assert os.path.exists(acoustic_g2p_model_path)
    assert os.path.exists(output_directory)


def test_train_and_align_basic_speaker_dict(
    multilingual_ipa_tg_corpus_dir,
    mfa_speaker_dict_path,
    generated_dir,
    temp_dir,
    basic_train_config_path,
    textgrid_output_model_path,
    english_mfa_phone_groups_path,
    english_mfa_rules_path,
    english_mfa_topology_path,
    bad_topology_path,
    db_setup,
):
    if os.path.exists(textgrid_output_model_path):
        os.remove(textgrid_output_model_path)
    output_directory = generated_dir.joinpath("ipa speaker output")
    with pytest.raises(PhoneGroupTopologyMismatchError):
        command = [
            "train",
            multilingual_ipa_tg_corpus_dir,
            mfa_speaker_dict_path,
            textgrid_output_model_path,
            "--config_path",
            basic_train_config_path,
            "-q",
            "--clean",
            "--no_debug",
            "--output_directory",
            output_directory,
            "--single_speaker",
            "--phone_groups_path",
            english_mfa_phone_groups_path,
            "--rules_path",
            english_mfa_rules_path,
            "--topology_path",
            bad_topology_path,
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
    command = [
        "train",
        multilingual_ipa_tg_corpus_dir,
        mfa_speaker_dict_path,
        textgrid_output_model_path,
        "--config_path",
        basic_train_config_path,
        "-q",
        "--clean",
        "--no_debug",
        "--output_directory",
        output_directory,
        "--single_speaker",
        "--phone_groups_path",
        english_mfa_phone_groups_path,
        "--rules_path",
        english_mfa_rules_path,
        "--topology_path",
        english_mfa_topology_path,
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
    assert os.path.exists(textgrid_output_model_path)
    assert os.path.exists(output_directory)
