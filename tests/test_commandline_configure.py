import os

import click.testing

from montreal_forced_aligner import config
from montreal_forced_aligner.command_line.mfa import mfa_cli


def test_configure(
    temp_dir,
    basic_corpus_dir,
    generated_dir,
    english_dictionary,
    basic_align_config_path,
    english_acoustic_model,
    global_config,
):
    path = config.generate_config_path()
    if os.path.exists(path):
        os.remove(path)
    command = [
        "configure",
        "--always_clean",
        "-t",
        temp_dir,
        "-j",
        "10",
        "--disable_mp",
        "--always_verbose",
        "-p",
        "test",
    ]
    command = [str(x) for x in command]
    click.testing.CliRunner().invoke(mfa_cli, command, catch_exceptions=False)
    assert os.path.exists(path)
    config.load_configuration()

    assert config.CURRENT_PROFILE_NAME == "test"
    assert config.NUM_JOBS == 10
    assert not config.USE_MP
    assert config.VERBOSE
    assert config.CLEAN

    command = ["configure", "--never_clean", "--enable_mp", "--never_verbose", "-p", "test"]
    click.testing.CliRunner().invoke(mfa_cli, command, catch_exceptions=False)

    assert os.path.exists(path)
    config.load_configuration()
    assert config.CURRENT_PROFILE_NAME == "test"
    assert config.USE_MP
    assert not config.VERBOSE
    assert not config.CLEAN

    config.CLEAN = True
    config.DEBUG = True
    config.VERBOSE = True
    config.USE_MP = False
    config.TEMPORARY_DIRECTORY = temp_dir
