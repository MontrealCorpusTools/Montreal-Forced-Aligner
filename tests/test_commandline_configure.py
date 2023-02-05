import os

import click.testing
import pytest

from montreal_forced_aligner.command_line.mfa import mfa_cli
from montreal_forced_aligner.config import generate_config_path


@pytest.mark.skip()
def test_configure(
    temp_dir,
    basic_corpus_dir,
    generated_dir,
    english_dictionary,
    basic_align_config_path,
    english_acoustic_model,
    global_config,
):
    path = generate_config_path()
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
    ]
    click.testing.CliRunner().invoke(mfa_cli, command, catch_exceptions=False)
    assert os.path.exists(path)
    global_config.load()

    assert global_config.current_profile_name == "test"
    assert global_config.current_profile.num_jobs == 10
    assert not global_config.current_profile.use_mp
    assert global_config.current_profile.verbose
    assert global_config.current_profile.clean

    command = ["configure", "--never_clean", "--enable_mp", "--never_verbose"]
    click.testing.CliRunner().invoke(mfa_cli, command, catch_exceptions=False)

    assert os.path.exists(path)
    global_config.load()
    assert global_config.current_profile_name == "test"
    assert global_config.current_profile.use_mp
    assert not global_config.current_profile.verbose
    assert not global_config.current_profile.clean

    global_config.clean = True
    global_config.debug = True
    global_config.verbose = True
    global_config.use_mp = False
    global_config.temporary_directory = temp_dir
    global_config.save()
