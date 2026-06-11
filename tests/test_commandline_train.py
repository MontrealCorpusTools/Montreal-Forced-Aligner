import os

import click.testing
import pytest

from montreal_forced_aligner.command_line.mfa import mfa_cli
from montreal_forced_aligner.exceptions import PhoneGroupTopologyMismatchError
from montreal_forced_aligner.models import MfaAlignmentModel


def test_train_acoustic_hf_output(
    multilingual_ipa_tg_corpus_dir,
    mfa_speaker_dict_path,
    generated_dir,
    temp_dir,
    basic_train_config_path,
    english_mfa_phone_groups_path,
    english_mfa_rules_path,
    english_mfa_topology_path,
    train_metadata_path,
    bad_topology_path,
    db_setup,
):
    output_model = generated_dir.joinpath("hg_model")
    command = [
        "train",
        multilingual_ipa_tg_corpus_dir,
        mfa_speaker_dict_path,
        output_model,
        "--config_path",
        basic_train_config_path,
        "-q",
        "--clean",
        "--no_debug",
        "--single_speaker",
        "--phone_groups_path",
        english_mfa_phone_groups_path,
        "--rules_path",
        english_mfa_rules_path,
        "--topology_path",
        english_mfa_topology_path,
        "--metadata_path",
        train_metadata_path,
        "--use_postgres",
        "--random_starts",
        "1",
        "--num_iterations",
        "3",
    ]
    command = [str(x) for x in command]
    result = click.testing.CliRunner().invoke(mfa_cli, command, catch_exceptions=True)
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value
    assert os.path.exists(output_model)

    model = MfaAlignmentModel(output_model, output_model)
    model.validate()
    print(model.model_card_path)
    with open(model.model_card_path, "r", encoding="utf8") as f:
        text = f.read()
        assert "[More Information Needed]" not in text


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
        result = click.testing.CliRunner().invoke(mfa_cli, command, catch_exceptions=True)
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
    result = click.testing.CliRunner().invoke(mfa_cli, command, catch_exceptions=True)
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value
    assert os.path.exists(textgrid_output_model_path)
    assert os.path.exists(output_directory)
