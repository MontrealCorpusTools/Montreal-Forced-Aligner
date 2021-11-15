"""
Feature generation functions
----------------------------

"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import TYPE_CHECKING, Dict, List, Union

from ..helper import load_scp, make_safe
from ..multiprocessing import run_mp, run_non_mp
from ..utils import thirdparty_binary

if TYPE_CHECKING:
    SpeakerCharacterType = Union[str, int]
    from ..abc import MetaDict
    from ..corpus import Corpus

__all__ = ["mfcc", "compute_vad", "calc_cmvn", "mfcc_func", "compute_vad_func"]


def mfcc_func(
    log_path: str,
    dictionaries: List[str],
    feats_scp_paths: Dict[str, str],
    lengths_paths: Dict[str, str],
    segment_paths: Dict[str, str],
    wav_paths: Dict[str, str],
    mfcc_options: MetaDict,
) -> None:
    """
    Multiprocessing function for generating MFCC features

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: List[str]
        List of dictionary names
    feats_scp_paths: Dict[str, str]
        Dictionary of feature scp files per dictionary name
    lengths_paths: Dict[str, str]
        Dictionary of feature lengths files per dictionary name
    segment_paths: Dict[str, str]
        Dictionary of segment scp files per dictionary name
    wav_paths: Dict[str, str]
        Dictionary of sound file scp files per dictionary name
    mfcc_options: :class:`~montreal_forced_aligner.abc.MetaDict`
        Options for MFCC generation
    """
    with open(log_path, "w") as log_file:
        for dict_name in dictionaries:
            mfcc_base_command = [thirdparty_binary("compute-mfcc-feats"), "--verbose=2"]
            raw_ark_path = feats_scp_paths[dict_name].replace(".scp", ".ark")
            for k, v in mfcc_options.items():
                mfcc_base_command.append(f"--{k.replace('_', '-')}={make_safe(v)}")
            if os.path.exists(segment_paths[dict_name]):
                mfcc_base_command += ["ark:-", "ark:-"]
                seg_proc = subprocess.Popen(
                    [
                        thirdparty_binary("extract-segments"),
                        f"scp,p:{wav_paths[dict_name]}",
                        segment_paths[dict_name],
                        "ark:-",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=os.environ,
                )
                comp_proc = subprocess.Popen(
                    mfcc_base_command,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    stdin=seg_proc.stdout,
                    env=os.environ,
                )
            else:
                mfcc_base_command += [f"scp,p:{wav_paths[dict_name]}", "ark:-"]
                comp_proc = subprocess.Popen(
                    mfcc_base_command, stdout=subprocess.PIPE, stderr=log_file, env=os.environ
                )
            copy_proc = subprocess.Popen(
                [
                    thirdparty_binary("copy-feats"),
                    "--compress=true",
                    "ark:-",
                    f"ark,scp:{raw_ark_path},{feats_scp_paths[dict_name]}",
                ],
                stdin=comp_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            copy_proc.communicate()

            utt_lengths_proc = subprocess.Popen(
                [
                    thirdparty_binary("feat-to-len"),
                    f"scp:{feats_scp_paths[dict_name]}",
                    f"ark,t:{lengths_paths[dict_name]}",
                ],
                stderr=log_file,
                env=os.environ,
            )
            utt_lengths_proc.communicate()


def mfcc(corpus: Corpus) -> None:
    """
    Multiprocessing function that converts sound files into MFCCs

    See http://kaldi-asr.org/doc/feat.html and
    http://kaldi-asr.org/doc/compute-mfcc-feats_8cc.html for more details on how
    MFCCs are computed.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/make_mfcc.sh
    for the bash script this function was based on.

    Parameters
    ----------
    corpus : :class:`~montreal_forced_aligner.corpus.Corpus`
        Corpus to generate MFCC features for
    """
    log_directory = os.path.join(corpus.split_directory, "log")
    os.makedirs(log_directory, exist_ok=True)

    jobs = [job.mfcc_arguments(corpus) for job in corpus.jobs]
    if corpus.use_mp:
        run_mp(mfcc_func, jobs, log_directory)
    else:
        run_non_mp(mfcc_func, jobs, log_directory)


def calc_cmvn(corpus: Corpus) -> None:
    """
    Calculate CMVN statistics for speakers

    Parameters
    ----------
    corpus : :class:`~montreal_forced_aligner.corpus.Corpus`
        Corpus to run CMVN calculation
    """
    spk2utt = os.path.join(corpus.output_directory, "spk2utt.scp")
    feats = os.path.join(corpus.output_directory, "feats.scp")
    cmvn_directory = os.path.join(corpus.features_directory, "cmvn")
    os.makedirs(cmvn_directory, exist_ok=True)
    cmvn_ark = os.path.join(cmvn_directory, "cmvn.ark")
    cmvn_scp = os.path.join(cmvn_directory, "cmvn.scp")
    log_path = os.path.join(cmvn_directory, "cmvn.log")
    with open(log_path, "w") as logf:
        subprocess.call(
            [
                thirdparty_binary("compute-cmvn-stats"),
                f"--spk2utt=ark:{spk2utt}",
                f"scp:{feats}",
                f"ark,scp:{cmvn_ark},{cmvn_scp}",
            ],
            stderr=logf,
            env=os.environ,
        )
    shutil.copy(cmvn_scp, os.path.join(corpus.output_directory, "cmvn.scp"))
    for s, cmvn in load_scp(cmvn_scp).items():
        corpus.speakers[s].cmvn = cmvn
    corpus.split()


def compute_vad_func(
    log_path: str,
    dictionaries: List[str],
    feats_scp_paths: Dict[str, str],
    vad_scp_paths: Dict[str, str],
    vad_options: MetaDict,
) -> None:
    """
    Multiprocessing function to compute voice activity detection

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: List[str]
        List of dictionary names
    feats_scp_paths: Dict[str, str]
        PronunciationDictionary of feature scp files per dictionary name
    vad_scp_paths: Dict[str, str]
        PronunciationDictionary of vad scp files per dictionary name
    vad_options: :class:`~montreal_forced_aligner.abc.MetaDict`
        Options for VAD
    """
    with open(log_path, "w") as log_file:
        for dict_name in dictionaries:
            feats_scp_path = feats_scp_paths[dict_name]
            vad_scp_path = vad_scp_paths[dict_name]
            vad_proc = subprocess.Popen(
                [
                    thirdparty_binary("compute-vad"),
                    f"--vad-energy-mean-scale={vad_options['energy_mean_scale']}",
                    f"--vad-energy-threshold={vad_options['energy_threshold']}",
                    "scp:" + feats_scp_path,
                    f"ark,t:{vad_scp_path}",
                ],
                stderr=log_file,
                env=os.environ,
            )
            vad_proc.communicate()


def compute_vad(corpus: Corpus) -> None:
    """
    Compute VAD for a corpus

    Parameters
    ----------
    corpus : :class:`~montreal_forced_aligner.corpus.Corpus`
        Corpus to compute VAD
    """
    log_directory = os.path.join(corpus.split_directory, "log")
    os.makedirs(log_directory, exist_ok=True)
    jobs = [x.vad_arguments(corpus) for x in corpus.jobs]
    if corpus.use_mp:
        run_mp(compute_vad_func, jobs, log_directory)
    else:
        run_non_mp(compute_vad_func, jobs, log_directory)
