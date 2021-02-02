import os
import shutil
import pytest

from montreal_forced_aligner.command_line.thirdparty import run_thirdparty, ArgumentError
from montreal_forced_aligner.thirdparty.download import download_binaries
from montreal_forced_aligner.thirdparty.kaldi import validate_kaldi_binaries, collect_kaldi_binaries
from montreal_forced_aligner.config import TEMP_DIR
from montreal_forced_aligner.command_line.mfa import parser



def test_download():
    bin_dir = os.path.join(TEMP_DIR, 'thirdparty', 'bin')
    if os.path.exists(bin_dir):
        shutil.rmtree(bin_dir, ignore_errors=True)

    assert not validate_kaldi_binaries()

    command = ['thirdparty', 'download']
    args, unknown = parser.parse_known_args(command)
    run_thirdparty(args)

    assert validate_kaldi_binaries()


def test_collect():
    bin_dir = os.path.join(TEMP_DIR, 'thirdparty', 'bin')
    backup_dir = os.path.join(TEMP_DIR, 'thirdparty', 'bin_backup')
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    assert validate_kaldi_binaries()

    shutil.move(bin_dir, backup_dir)

    assert not validate_kaldi_binaries()

    command = ['thirdparty', 'kaldi', backup_dir]
    args, unknown = parser.parse_known_args(command)
    run_thirdparty(args)

    assert validate_kaldi_binaries()


def test_validate():
    command = ['thirdparty', 'validate']
    args, unknown = parser.parse_known_args(command)
    run_thirdparty(args)
