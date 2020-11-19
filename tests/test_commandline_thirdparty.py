import os
import shutil

from montreal_forced_aligner.thirdparty.download import download_binaries
from montreal_forced_aligner.thirdparty.kaldi import validate_kaldi_binaries, collect_kaldi_binaries
from montreal_forced_aligner.thirdparty.ngram import validate_ngram_binaries, collect_ngram_binaries
from montreal_forced_aligner.thirdparty.phonetisaurus import validate_phonetisaurus_binaries, collect_phonetisaurus_binaries
from montreal_forced_aligner.config import TEMP_DIR


def test_download():
    bin_dir = os.path.join(TEMP_DIR, 'thirdparty', 'bin')
    if os.path.exists(bin_dir):
        shutil.rmtree(bin_dir, ignore_errors=True)

    assert not validate_kaldi_binaries()
    assert not validate_phonetisaurus_binaries()
    assert not validate_ngram_binaries()

    download_binaries()

    assert validate_kaldi_binaries()
    assert validate_ngram_binaries()
    assert validate_phonetisaurus_binaries()


def test_collect():
    bin_dir = os.path.join(TEMP_DIR, 'thirdparty', 'bin')
    backup_dir = os.path.join(TEMP_DIR, 'thirdparty', 'bin_backup')

    assert validate_kaldi_binaries()
    assert validate_ngram_binaries()
    assert validate_phonetisaurus_binaries()

    shutil.move(bin_dir, backup_dir)

    assert not validate_kaldi_binaries()
    assert not validate_phonetisaurus_binaries()
    assert not validate_ngram_binaries()

    collect_kaldi_binaries(backup_dir)
    collect_ngram_binaries(backup_dir)
    collect_phonetisaurus_binaries(backup_dir)

    assert validate_kaldi_binaries()
    assert validate_ngram_binaries()
    assert validate_phonetisaurus_binaries()
