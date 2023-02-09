"""Classes for configuring feature generation"""
from __future__ import annotations

import io
import logging
import math
import os
import re
import subprocess
import typing
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Union

import dataclassy
import numba
import numpy as np
import sqlalchemy
from numba import njit
from scipy.sparse import csr_matrix
from sqlalchemy.orm import Session, joinedload

from montreal_forced_aligner.abc import KaldiFunction
from montreal_forced_aligner.config import IVECTOR_DIMENSION, PLDA_DIMENSION
from montreal_forced_aligner.data import M_LOG_2PI, MfaArguments
from montreal_forced_aligner.db import Job, Utterance
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.utils import read_feats, thirdparty_binary

if TYPE_CHECKING:
    SpeakerCharacterType = Union[str, int]
    from montreal_forced_aligner.abc import MetaDict


__all__ = [
    "FeatureConfigMixin",
    "VadConfigMixin",
    "IvectorConfigMixin",
    "CalcFmllrFunction",
    "ComputeVadFunction",
    "VadArguments",
    "MfccFunction",
    "MfccArguments",
    "CalcFmllrArguments",
    "ExtractIvectorsFunction",
    "ExtractIvectorsArguments",
    "PldaModel",
    "plda_distance",
    "plda_log_likelihood",
    "score_plda",
    "compute_transform_process",
]

logger = logging.getLogger("mfa")


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class VadArguments(MfaArguments):
    """Arguments for :class:`~montreal_forced_aligner.corpus.features.ComputeVadFunction`"""

    feats_scp_path: str
    vad_scp_path: str
    vad_options: MetaDict


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class MfccArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.corpus.features.MfccFunction`
    """

    data_directory: str
    mfcc_options: MetaDict
    pitch_options: MetaDict


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class FinalFeatureArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.corpus.features.FinalFeatureFunction`
    """

    data_directory: str
    uses_cmvn: bool
    voiced_only: bool
    subsample_feats: int


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class PitchArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.corpus.features.MfccFunction`
    """

    data_directory: str
    pitch_options: MetaDict


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class PitchRangeArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.corpus.features.MfccFunction`
    """

    data_directory: str
    pitch_options: MetaDict


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class CalcFmllrArguments(MfaArguments):
    """Arguments for :class:`~montreal_forced_aligner.corpus.features.CalcFmllrFunction`"""

    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ali_paths: Dict[str, str]
    ali_model_path: str
    model_path: str
    spk2utt_paths: Dict[str, str]
    trans_paths: Dict[str, str]
    fmllr_options: MetaDict


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class ExtractIvectorsArguments(MfaArguments):
    """Arguments for :class:`~montreal_forced_aligner.corpus.features.ExtractIvectorsFunction`"""

    ivector_options: MetaDict
    ie_path: str
    ivectors_scp_path: str
    dubm_path: str


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class ExportIvectorsArguments(MfaArguments):
    """Arguments for :class:`~montreal_forced_aligner.corpus.features.ExportIvectorsFunction`"""

    use_xvector: bool


def feature_make_safe(value: Any) -> str:
    """
    Transform an arbitrary value into a string

    Parameters
    ----------
    value: Any
        Value to make safe

    Returns
    -------
    str
        Safe value
    """
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def compute_mfcc_process(
    log_file: io.FileIO,
    wav_path: str,
    segments: typing.Union[str, subprocess.Popen, subprocess.PIPE],
    mfcc_options: MetaDict,
    min_length=0.1,
) -> subprocess.Popen:
    """
    Construct processes for computing features

    Parameters
    ----------
    log_file: io.FileIO
        File for logging stderr
    wav_path: str
        Wav scp to use
    segments: str
        Segments scp to use
    mfcc_options: dict[str, Any]
        Options for computing MFCC features
    min_length: float
        Minimum length of segments in seconds
    no_logging: bool
        Flag for logging progress information to log_file rather than a subprocess pipe

    Returns
    -------
    subprocess.Popen
        MFCC process
    """
    mfcc_base_command = [thirdparty_binary("compute-mfcc-feats")]
    for k, v in mfcc_options.items():
        mfcc_base_command.append(f"--{k.replace('_', '-')}={feature_make_safe(v)}")
    if isinstance(segments, str) and os.path.exists(segments):
        mfcc_base_command += ["ark:-", "ark,t:-"]
        seg_proc = subprocess.Popen(
            [
                thirdparty_binary("extract-segments"),
                f"--min-segment-length={min_length}",
                f"scp:{wav_path}",
                segments,
                "ark:-",
            ],
            stdout=subprocess.PIPE,
            stderr=log_file,
            env=os.environ,
        )
        mfcc_proc = subprocess.Popen(
            mfcc_base_command,
            stdout=subprocess.PIPE,
            stderr=log_file,
            stdin=seg_proc.stdout,
            env=os.environ,
        )
    elif isinstance(segments, subprocess.Popen):
        mfcc_base_command += ["ark,s,cs:-", "ark,t:-"]
        mfcc_proc = subprocess.Popen(
            mfcc_base_command,
            stdout=subprocess.PIPE,
            stderr=log_file,
            stdin=segments.stdout,
            env=os.environ,
        )
    elif segments == subprocess.PIPE:
        mfcc_base_command += ["ark,s,cs:-", "ark,t:-"]
        mfcc_proc = subprocess.Popen(
            mfcc_base_command,
            stdout=subprocess.PIPE,
            stderr=log_file,
            stdin=segments,
            env=os.environ,
        )
    else:
        mfcc_base_command += [f"scp,p:{wav_path}", "ark:-"]
        mfcc_proc = subprocess.Popen(
            mfcc_base_command,
            stdout=subprocess.PIPE,
            stderr=log_file,
            env=os.environ,
        )

    return mfcc_proc


def compute_pitch_process(
    log_file: io.FileIO,
    wav_path: str,
    segments: typing.Union[str, subprocess.Popen, subprocess.PIPE],
    pitch_options: MetaDict,
    min_length=0.1,
) -> subprocess.Popen:
    """
    Construct processes for computing features

    Parameters
    ----------
    log_file: io.FileIO
        File for logging stderr
    wav_path: str
        Wav scp to use
    segments: str
        Segments scp to use
    mfcc_options: dict[str, Any]
        Options for computing MFCC features
    pitch_options: dict[str, Any]
        Options for computing pitch features
    min_length: float
        Minimum length of segments in seconds
    no_logging: bool
        Flag for logging progress information to log_file rather than a subprocess pipe

    Returns
    -------
    subprocess.Popen
        Pitch process
    """
    use_pitch = pitch_options.pop("use-pitch")
    use_voicing = pitch_options.pop("use-voicing")
    use_delta_pitch = pitch_options.pop("use-delta-pitch")
    normalize = pitch_options.pop("normalize", True)
    pitch_command = [
        thirdparty_binary("compute-and-process-kaldi-pitch-feats"),
    ]
    for k, v in pitch_options.items():
        pitch_command.append(f"--{k.replace('_', '-')}={feature_make_safe(v)}")
        if k == "delta-pitch":
            pitch_command.append(f"--delta-pitch-noise-stddev={feature_make_safe(v)}")
    if use_pitch:
        if normalize:
            pitch_command.append("--add-normalized-log-pitch=true")
        else:
            pitch_command.append("--add-raw-log-pitch=true")
    else:
        pitch_command.append("--add-normalized-log-pitch=false")
        pitch_command.append("--add-raw-log-pitch=false")
    if use_delta_pitch:
        pitch_command.append("--add-delta-pitch=true")
        pitch_command.append("--add-pov-feature=true")
    else:
        pitch_command.append("--add-delta-pitch=false")
        if use_voicing:
            pitch_command.append("--add-pov-feature=true")
        else:
            pitch_command.append("--add-pov-feature=false")

    if isinstance(segments, str) and os.path.exists(segments):
        pitch_command += ["ark:-", "ark,t:-"]
        seg_proc = subprocess.Popen(
            [
                thirdparty_binary("extract-segments"),
                f"--min-segment-length={min_length}",
                f"scp:{wav_path}",
                segments,
                "ark:-",
            ],
            stdout=subprocess.PIPE,
            stderr=log_file,
            env=os.environ,
        )
        pitch_proc = subprocess.Popen(
            pitch_command,
            stdout=subprocess.PIPE,
            stderr=log_file,
            stdin=seg_proc.stdout,
            env=os.environ,
        )
    elif isinstance(segments, subprocess.Popen):
        pitch_command += ["ark:-", "ark,t:-"]
        pitch_proc = subprocess.Popen(
            pitch_command,
            stdout=subprocess.PIPE,
            stderr=log_file,
            stdin=segments.stdout,
            env=os.environ,
        )
    elif segments == subprocess.PIPE:
        pitch_command += ["ark:-", "ark,t:-"]
        pitch_proc = subprocess.Popen(
            pitch_command,
            stdout=subprocess.PIPE,
            stderr=log_file,
            stdin=segments,
            env=os.environ,
        )
    else:
        pitch_command += [f"scp,p:{wav_path}", "ark,t:-"]
        pitch_proc = subprocess.Popen(
            pitch_command,
            stdout=subprocess.PIPE,
            stderr=log_file,
            env=os.environ,
        )
    return pitch_proc


def compute_transform_process(
    log_file: io.FileIO,
    feat_proc: typing.Union[subprocess.Popen, str],
    utt2spk_path: str,
    lda_mat_path: typing.Optional[str],
    fmllr_path: typing.Optional[str],
    lda_options: MetaDict,
) -> subprocess.Popen:
    """
    Construct feature transformation process

    Parameters
    ----------
    log_file: io.FileIO
        File for logging stderr
    feat_proc: subprocess.Popen
        Feature generation process
    utt2spk_path: str
        Utterance to speaker SCP file path
    cmvn_path: str
        CMVN SCP file path
    lda_mat_path: str
        LDA matrix file path
    fmllr_path: str
        fMLLR transform file path
    lda_options: dict[str, Any]
        Options for LDA

    Returns
    -------
    subprocess.Popen
        Processing for transforming features
    """
    if isinstance(feat_proc, str):
        feat_input = f"ark,s,cs:{feat_proc}"
        use_stdin = False
    else:
        feat_input = "ark,s,cs:-"
        use_stdin = True
    if lda_mat_path is not None:
        splice_proc = subprocess.Popen(
            [
                "splice-feats",
                f'--left-context={lda_options["splice_left_context"]}',
                f'--right-context={lda_options["splice_right_context"]}',
                feat_input,
                "ark:-",
            ],
            env=os.environ,
            stdin=feat_proc.stdout if use_stdin else None,
            stdout=subprocess.PIPE,
            stderr=log_file,
        )
        delta_proc = subprocess.Popen(
            ["transform-feats", lda_mat_path, "ark,s,cs:-", "ark:-"],
            env=os.environ,
            stdin=splice_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=log_file,
        )
    else:
        delta_proc = subprocess.Popen(
            ["add-deltas", feat_input, "ark:-"],
            env=os.environ,
            stdin=feat_proc.stdout if use_stdin else None,
            stdout=subprocess.PIPE,
            stderr=log_file,
        )
    if fmllr_path is None:
        return delta_proc

    fmllr_proc = subprocess.Popen(
        [
            "transform-feats",
            f"--utt2spk=ark:{utt2spk_path}",
            f"ark:{fmllr_path}",
            "ark,s,cs:-",
            "ark,t:-",
        ],
        env=os.environ,
        stdin=delta_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=log_file,
    )
    return fmllr_proc


class MfccFunction(KaldiFunction):
    """
    Multiprocessing function for generating MFCC features

    See Also
    --------
    :meth:`.AcousticCorpusMixin.mfcc`
        Main function that calls this function in parallel
    :meth:`.AcousticCorpusMixin.mfcc_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`compute-mfcc-feats`
        Relevant Kaldi binary
    :kaldi_src:`extract-segments`
        Relevant Kaldi binary
    :kaldi_src:`copy-feats`
        Relevant Kaldi binary
    :kaldi_src:`feat-to-len`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.corpus.features.MfccArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(r"^LOG.* Processed (?P<num_utterances>\d+) utterances")

    def __init__(self, args: MfccArguments):
        super().__init__(args)
        self.data_directory = args.data_directory
        self.pitch_options = args.pitch_options
        self.mfcc_options = args.mfcc_options

    def _run(self) -> typing.Generator[int]:
        """Run the function"""
        with Session(self.db_engine) as session, mfa_open(self.log_path, "w") as log_file:
            job: Job = session.get(Job, self.job_name)
            feats_scp_path = job.construct_path(self.data_directory, "feats", "scp")
            pitch_scp_path = job.construct_path(self.data_directory, "pitch", "scp")
            segments_scp_path = job.construct_path(self.data_directory, "segments", "scp")
            wav_path = job.construct_path(self.data_directory, "wav", "scp")
            raw_ark_path = job.construct_path(self.data_directory, "feats", "ark")
            raw_pitch_ark_path = job.construct_path(self.data_directory, "pitch", "ark")
            if os.path.exists(raw_ark_path):
                return
            min_length = 0.1
            seg_proc = subprocess.Popen(
                [
                    thirdparty_binary("extract-segments"),
                    f"--min-segment-length={min_length}",
                    f"scp:{wav_path}",
                    segments_scp_path,
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            mfcc_proc = compute_mfcc_process(
                log_file, wav_path, subprocess.PIPE, self.mfcc_options
            )
            mfcc_copy_proc = subprocess.Popen(
                [
                    thirdparty_binary("copy-feats"),
                    "--compress=true",
                    "ark:-",
                    f"ark,scp:{raw_ark_path},{feats_scp_path}",
                ],
                stdin=mfcc_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            use_pitch = self.pitch_options["use-pitch"] or self.pitch_options["use-voicing"]
            if use_pitch:
                pitch_proc = compute_pitch_process(
                    log_file, wav_path, subprocess.PIPE, self.pitch_options
                )
                pitch_copy_proc = subprocess.Popen(
                    [
                        thirdparty_binary("copy-feats"),
                        "--compress=true",
                        "ark:-",
                        f"ark,scp:{raw_pitch_ark_path},{pitch_scp_path}",
                    ],
                    stdin=pitch_proc.stdout,
                    stderr=log_file,
                    env=os.environ,
                )
            for line in seg_proc.stdout:
                mfcc_proc.stdin.write(line)
                mfcc_proc.stdin.flush()
                if use_pitch:
                    pitch_proc.stdin.write(line)
                    pitch_proc.stdin.flush()
                if re.search(rb"\d+-\d+ ", line):
                    yield 1
            mfcc_proc.stdin.close()
            if use_pitch:
                pitch_proc.stdin.close()
            mfcc_proc.wait()
            if use_pitch:
                pitch_proc.wait()
            self.check_call(mfcc_copy_proc)
            if use_pitch:
                self.check_call(pitch_copy_proc)


class FinalFeatureFunction(KaldiFunction):
    """
    Multiprocessing function for generating MFCC features

    See Also
    --------
    :meth:`.AcousticCorpusMixin.mfcc`
        Main function that calls this function in parallel
    :meth:`.AcousticCorpusMixin.mfcc_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`compute-mfcc-feats`
        Relevant Kaldi binary
    :kaldi_src:`extract-segments`
        Relevant Kaldi binary
    :kaldi_src:`copy-feats`
        Relevant Kaldi binary
    :kaldi_src:`feat-to-len`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.corpus.features.MfccArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(r"^LOG.* Processed (?P<num_utterances>\d+) utterances")

    def __init__(self, args: FinalFeatureArguments):
        super().__init__(args)
        self.data_directory = args.data_directory
        self.voiced_only = args.voiced_only
        self.uses_cmvn = args.uses_cmvn
        self.subsample_feats = args.subsample_feats

    def _run(self) -> typing.Generator[int]:
        """Run the function"""
        with Session(self.db_engine) as session, mfa_open(self.log_path, "w") as log_file:
            job: Job = session.get(Job, self.job_name)
            feats_scp_path = job.construct_path(self.data_directory, "feats", "scp")
            temp_scp_path = job.construct_path(self.data_directory, "final_features", "scp")
            utt2spk_path = job.construct_path(self.data_directory, "utt2spk", "scp")
            cmvn_scp_path = job.construct_path(self.data_directory, "cmvn", "scp")
            pitch_scp_path = job.construct_path(self.data_directory, "pitch", "scp")
            pitch_ark_path = job.construct_path(self.data_directory, "pitch", "ark")
            vad_scp_path = job.construct_path(self.data_directory, "vad", "scp")
            raw_ark_path = job.construct_path(self.data_directory, "feats", "ark")
            temp_ark_path = job.construct_path(self.data_directory, "final_features", "ark")
            if os.path.exists(cmvn_scp_path):
                cmvn_proc = subprocess.Popen(
                    [
                        thirdparty_binary("apply-cmvn"),
                        f"--utt2spk=ark:{utt2spk_path}",
                        f"scp:{cmvn_scp_path}",
                        f"scp:{feats_scp_path}",
                        "ark:-",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=os.environ,
                )
            else:
                cmvn_proc = subprocess.Popen(
                    [
                        thirdparty_binary("apply-cmvn-sliding"),
                        "--norm-vars=false",
                        "--center=true",
                        "--cmn-window=300",
                        f"scp:{feats_scp_path}",
                        "ark:-",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=os.environ,
                )
            if os.path.exists(pitch_scp_path):
                paste_proc = subprocess.Popen(
                    [
                        thirdparty_binary("paste-feats"),
                        "--length-tolerance=2",
                        "ark:-",
                        f"scp:{pitch_scp_path}",
                        "ark:-",
                    ],
                    stdin=cmvn_proc.stdout,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=os.environ,
                )
            else:
                paste_proc = cmvn_proc
            if self.voiced_only and os.path.exists(vad_scp_path):
                voiced_proc = subprocess.Popen(
                    [
                        thirdparty_binary("select-voiced-frames"),
                        "ark:-",
                        f"scp:{vad_scp_path}",
                        "ark:-",
                    ],
                    stdin=paste_proc.stdout,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=os.environ,
                )
                if self.subsample_feats:
                    final_proc = subprocess.Popen(
                        [
                            thirdparty_binary("subsample-feats"),
                            f"--n={self.subsample_feats}",
                            "ark:-",
                            "ark:-",
                        ],
                        stdin=voiced_proc.stdout,
                        stdout=subprocess.PIPE,
                        stderr=log_file,
                        env=os.environ,
                    )
                else:
                    final_proc = voiced_proc
            else:
                final_proc = paste_proc
            copy_proc = subprocess.Popen(
                [
                    thirdparty_binary("copy-feats"),
                    "--compress=true",
                    "ark:-",
                    f"ark,scp:{temp_ark_path},{temp_scp_path}",
                ],
                stdin=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )

            for line in final_proc.stdout:
                copy_proc.stdin.write(line)
                copy_proc.stdin.flush()
                if re.search(rb"\d+-\d+ ", line):
                    yield 1
            copy_proc.stdin.close()
            self.check_call(copy_proc)
            os.remove(raw_ark_path)
            os.remove(feats_scp_path)
            os.rename(temp_scp_path, feats_scp_path)
            if os.path.exists(pitch_scp_path):
                os.remove(pitch_scp_path)
                os.remove(pitch_ark_path)


class PitchFunction(KaldiFunction):
    """
    Multiprocessing function for generating MFCC features

    See Also
    --------
    :meth:`.AcousticCorpusMixin.mfcc`
        Main function that calls this function in parallel
    :meth:`.AcousticCorpusMixin.mfcc_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`compute-mfcc-feats`
        Relevant Kaldi binary
    :kaldi_src:`extract-segments`
        Relevant Kaldi binary
    :kaldi_src:`copy-feats`
        Relevant Kaldi binary
    :kaldi_src:`feat-to-len`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.corpus.features.MfccArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(r"^LOG.* Processed (?P<num_utterances>\d+) utterances")

    def __init__(self, args: PitchArguments):
        super().__init__(args)
        self.data_directory = args.data_directory
        self.pitch_options = args.pitch_options

    def _run(self) -> typing.Generator[int]:
        """Run the function"""
        with Session(self.db_engine) as session, mfa_open(self.log_path, "w") as log_file:
            job: Job = session.get(Job, self.job_name)

            feats_scp_path = job.construct_path(self.data_directory, "pitch", "scp")
            raw_ark_path = job.construct_path(self.data_directory, "pitch", "ark")
            wav_path = job.construct_path(self.data_directory, "wav", "scp")
            segments_path = job.construct_path(self.data_directory, "segments", "scp")
            if os.path.exists(raw_ark_path):
                return
            copy_proc = subprocess.Popen(
                [
                    thirdparty_binary("copy-feats"),
                    "--compress=true",
                    "ark,t:-",
                    f"ark,scp:{raw_ark_path},{feats_scp_path}",
                ],
                stdin=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )

            pitch_proc = compute_pitch_process(
                log_file, wav_path, segments_path, self.pitch_options
            )
            for line in pitch_proc.stdout:
                copy_proc.stdin.write(line)
                copy_proc.stdin.flush()
                if re.match(rb"^\d+-", line):
                    yield 1
            pitch_proc.wait()
            copy_proc.stdin.close()
            self.check_call(copy_proc)


class PitchRangeFunction(KaldiFunction):
    """
    Multiprocessing function for generating MFCC features

    See Also
    --------
    :meth:`.AcousticCorpusMixin.mfcc`
        Main function that calls this function in parallel
    :meth:`.AcousticCorpusMixin.mfcc_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`compute-mfcc-feats`
        Relevant Kaldi binary
    :kaldi_src:`extract-segments`
        Relevant Kaldi binary
    :kaldi_src:`copy-feats`
        Relevant Kaldi binary
    :kaldi_src:`feat-to-len`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.corpus.features.MfccArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(r"^LOG.* Processed (?P<num_utterances>\d+) utterances")

    def __init__(self, args: PitchRangeArguments):
        super().__init__(args)
        self.data_directory = args.data_directory
        self.pitch_options = args.pitch_options

    def _run(self) -> typing.Generator[int]:
        """Run the function"""
        with Session(self.db_engine) as session, mfa_open(self.log_path, "w") as log_file:
            job: Job = session.get(Job, self.job_name)
            wav_path = job.construct_path(self.data_directory, "wav", "scp")
            segment_path = job.construct_path(self.data_directory, "segments", "scp")
            min_length = 0.1
            seg_proc = subprocess.Popen(
                [
                    thirdparty_binary("extract-segments"),
                    f"--min-segment-length={min_length}",
                    f"scp:{wav_path}",
                    segment_path,
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            pitch_command = [
                thirdparty_binary("compute-kaldi-pitch-feats"),
            ]
            for k, v in self.pitch_options.items():
                if k in {"use-pitch", "use-voicing", "normalize"}:
                    continue
                pitch_command.append(f"--{k.replace('_', '-')}={feature_make_safe(v)}")
            pitch_command += ["ark:-", "ark,t:-"]
            pitch_proc = subprocess.Popen(
                pitch_command,
                stdout=subprocess.PIPE,
                stdin=seg_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            current_speaker = None
            pitch_points = []
            for ids, pitch_features in read_feats(pitch_proc, raw_id=True):
                speaker_id, utt_id = ids.split("-")
                speaker_id = int(speaker_id)
                if current_speaker is None:
                    current_speaker = speaker_id
                if current_speaker != speaker_id:
                    pitch_points = np.array(pitch_points)
                    mean_f0 = np.mean(pitch_points)
                    min_f0 = mean_f0 / 2
                    max_f0 = mean_f0 * 2
                    yield current_speaker, max(min_f0, 50), min(max_f0, 1500)
                    pitch_points = []
                    current_speaker = speaker_id
                indices = np.where(pitch_features[:, 0] > 0.5)
                pitch_points.extend(pitch_features[indices[0], 1])
            self.check_call(pitch_proc)


class ComputeVadFunction(KaldiFunction):
    """
    Multiprocessing function to compute voice activity detection

    See Also
    --------
    :meth:`.AcousticCorpusMixin.compute_vad`
        Main function that calls this function in parallel
    :meth:`.AcousticCorpusMixin.compute_vad_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`compute-vad`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.corpus.features.VadArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(
        r"^LOG.*processed (?P<done>\d+) utterances.*(?P<no_feats>\d+) had.*(?P<unvoiced>\d+) were.*"
    )

    def __init__(self, args: VadArguments):
        super().__init__(args)
        self.feats_scp_path = args.feats_scp_path
        self.vad_scp_path = args.vad_scp_path
        self.vad_options = args.vad_options

    def _run(self) -> typing.Generator[typing.Tuple[int, int, int]]:
        """Run the function"""
        with mfa_open(self.log_path, "w") as log_file:
            feats_scp_path = self.feats_scp_path
            vad_scp_path = self.vad_scp_path
            vad_ark_path = self.vad_scp_path.replace(".scp", ".ark")
            vad_proc = subprocess.Popen(
                [
                    thirdparty_binary("compute-vad"),
                    f"--vad-energy-mean-scale={self.vad_options['energy_mean_scale']}",
                    f"--vad-energy-threshold={self.vad_options['energy_threshold']}",
                    f"scp:{feats_scp_path}",
                    f"ark,scp:{vad_ark_path},{vad_scp_path}",
                ],
                stderr=subprocess.PIPE,
                encoding="utf8",
                env=os.environ,
            )
            for line in vad_proc.stderr:
                log_file.write(line)
                m = self.progress_pattern.match(line.strip())
                if m:
                    yield int(m.group("done")), int(m.group("no_feats")), int(m.group("unvoiced"))
            self.check_call(vad_proc)


class CalcFmllrFunction(KaldiFunction):
    """
    Multiprocessing function for calculating fMLLR transforms

    See Also
    --------
    :meth:`.AcousticCorpusMixin.calc_fmllr`
        Main function that calls this function in parallel
    :meth:`.AcousticCorpusMixin.calc_fmllr_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`gmm-est-fmllr`
        Relevant Kaldi binary
    :kaldi_src:`gmm-est-fmllr-gpost`
        Relevant Kaldi binary
    :kaldi_src:`gmm-post-to-gpost`
        Relevant Kaldi binary
    :kaldi_src:`ali-to-post`
        Relevant Kaldi binary
    :kaldi_src:`weight-silence-post`
        Relevant Kaldi binary
    :kaldi_src:`compose-transforms`
        Relevant Kaldi binary
    :kaldi_src:`transform-feats`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.corpus.features.CalcFmllrArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(r"^LOG.*For speaker (?P<speaker>.*),.*$")
    memory_error_pattern = re.compile(
        r"^ERROR \(gmm-est-fmllr-gpost.*Failed to read vector from stream..*$"
    )

    def __init__(self, args: CalcFmllrArguments):
        super().__init__(args)
        self.dictionaries = args.dictionaries
        self.feature_strings = args.feature_strings
        self.ali_paths = args.ali_paths
        self.ali_model_path = args.ali_model_path
        self.model_path = args.model_path
        self.spk2utt_paths = args.spk2utt_paths
        self.trans_paths = args.trans_paths
        self.fmllr_options = args.fmllr_options

    def _run(self) -> typing.Generator[str]:
        """Run the function"""
        with mfa_open(self.log_path, "w") as log_file:
            for dict_id in self.dictionaries:
                while True:
                    feature_string = self.feature_strings[dict_id]
                    ali_path = self.ali_paths[dict_id]
                    spk2utt_path = self.spk2utt_paths[dict_id]
                    trans_path = self.trans_paths[dict_id]
                    initial = True
                    if os.path.exists(trans_path):
                        initial = False
                    post_proc = subprocess.Popen(
                        [thirdparty_binary("ali-to-post"), f"ark,s,cs:{ali_path}", "ark:-"],
                        stderr=log_file,
                        stdout=subprocess.PIPE,
                        env=os.environ,
                    )

                    weight_proc = subprocess.Popen(
                        [
                            thirdparty_binary("weight-silence-post"),
                            "0.0",
                            self.fmllr_options["silence_csl"],
                            self.ali_model_path,
                            "ark,s,cs:-",
                            "ark:-",
                        ],
                        stderr=log_file,
                        stdin=post_proc.stdout,
                        stdout=subprocess.PIPE,
                        env=os.environ,
                    )

                    temp_trans_path = trans_path + ".tmp"
                    if self.ali_model_path != self.model_path:
                        post_gpost_proc = subprocess.Popen(
                            [
                                thirdparty_binary("gmm-post-to-gpost"),
                                self.ali_model_path,
                                feature_string,
                                "ark,s,cs:-",
                                "ark:-",
                            ],
                            stderr=log_file,
                            stdin=weight_proc.stdout,
                            stdout=subprocess.PIPE,
                            env=os.environ,
                        )
                        est_proc = subprocess.Popen(
                            [
                                thirdparty_binary("gmm-est-fmllr-gpost"),
                                "--verbose=4",
                                f"--fmllr-update-type={self.fmllr_options['fmllr_update_type']}",
                                f"--spk2utt=ark:{spk2utt_path}",
                                self.model_path,
                                feature_string,
                                "ark,s,cs:-",
                                f"ark:{trans_path}",
                            ],
                            stderr=subprocess.PIPE,
                            encoding="utf8",
                            stdin=post_gpost_proc.stdout,
                            env=os.environ,
                        )

                    else:

                        if not initial:
                            temp_composed_trans_path = trans_path + ".cmp.tmp"
                            est_proc = subprocess.Popen(
                                [
                                    thirdparty_binary("gmm-est-fmllr"),
                                    "--verbose=4",
                                    f"--fmllr-update-type={self.fmllr_options['fmllr_update_type']}",
                                    f"--spk2utt=ark,s,cs:{spk2utt_path}",
                                    self.model_path,
                                    feature_string,
                                    "ark,s,cs:-",
                                    f"ark:{temp_trans_path}",
                                ],
                                stderr=subprocess.PIPE,
                                encoding="utf8",
                                stdin=weight_proc.stdout,
                                stdout=subprocess.PIPE,
                                env=os.environ,
                            )
                        else:
                            est_proc = subprocess.Popen(
                                [
                                    thirdparty_binary("gmm-est-fmllr"),
                                    "--verbose=4",
                                    f"--fmllr-update-type={self.fmllr_options['fmllr_update_type']}",
                                    f"--spk2utt=ark,s,cs:{spk2utt_path}",
                                    self.model_path,
                                    feature_string,
                                    "ark,s,cs:-",
                                    f"ark:{trans_path}",
                                ],
                                stderr=subprocess.PIPE,
                                encoding="utf8",
                                stdin=weight_proc.stdout,
                                env=os.environ,
                            )

                    for line in est_proc.stderr:
                        log_file.write(line)
                        m = self.progress_pattern.match(line.strip())
                        if m:
                            yield m.group("speaker")
                    try:
                        self.check_call(est_proc)
                        break
                    except KaldiProcessingError:  # Try to recover from Memory exception
                        with mfa_open(self.log_path, "r") as f:
                            for line in f:
                                if self.memory_error_pattern.match(line):
                                    os.remove(trans_path)
                                    break
                            else:
                                raise
                if not initial:
                    compose_proc = subprocess.Popen(
                        [
                            thirdparty_binary("compose-transforms"),
                            "--b-is-affine=true",
                            f"ark:{temp_trans_path}",
                            f"ark:{trans_path}",
                            f"ark:{temp_composed_trans_path}",
                        ],
                        stderr=log_file,
                        env=os.environ,
                    )
                    compose_proc.communicate()
                    self.check_call(compose_proc)

                    os.remove(trans_path)
                    os.remove(temp_trans_path)
                    os.rename(temp_composed_trans_path, trans_path)


class FeatureConfigMixin:
    """
    Class to store configuration information about MFCC generation

    Attributes
    ----------
    feature_type : str
        Feature type, defaults to "mfcc"
    use_energy : bool
        Flag for whether first coefficient should be used, defaults to False
    frame_shift : int
        number of milliseconds between frames, defaults to 10
    snip_edges : bool
        Flag for enabling Kaldi's snip edges, should be better time precision
    use_pitch : bool
        Flag for including pitch in features, defaults to False
    low_frequency : int
        Frequency floor
    high_frequency : int
        Frequency ceiling
    sample_frequency : int
        Sampling frequency
    allow_downsample : bool
        Flag for whether to allow downsampling, default is True
    allow_upsample : bool
        Flag for whether to allow upsampling, default is True
    uses_cmvn : bool
        Flag for whether to use CMVN, default is True
    uses_deltas : bool
        Flag for whether to use delta features, default is True
    uses_splices : bool
        Flag for whether to use splices and LDA transformations, default is False
    uses_speaker_adaptation : bool
        Flag for whether to use speaker adaptation, default is False
    fmllr_update_type : str
        Type of fMLLR estimation, defaults to "full"
    silence_weight : float
        Weight of silence in calculating LDA or fMLLR
    splice_left_context : int or None
        Number of frames to splice on the left for calculating LDA
    splice_right_context : int or None
        Number of frames to splice on the right for calculating LDA
    """

    def __init__(
        self,
        feature_type: str = "mfcc",
        use_energy: bool = False,
        frame_shift: int = 10,
        frame_length: int = 25,
        snip_edges: bool = True,
        low_frequency: int = 20,
        high_frequency: int = 7800,
        sample_frequency: int = 16000,
        allow_downsample: bool = True,
        allow_upsample: bool = True,
        dither: int = 1,
        energy_floor: float = 0,
        num_coefficients: int = 13,
        num_mel_bins: int = 23,
        cepstral_lifter: float = 22,
        preemphasis_coefficient: float = 0.97,
        uses_cmvn: bool = True,
        uses_deltas: bool = True,
        uses_splices: bool = False,
        uses_voiced: bool = False,
        adaptive_pitch_range: bool = False,
        uses_speaker_adaptation: bool = False,
        fmllr_update_type: str = "full",
        silence_weight: float = 0.0,
        splice_left_context: int = 3,
        splice_right_context: int = 3,
        use_pitch: bool = False,
        use_voicing: bool = False,
        use_delta_pitch: bool = False,
        min_f0: float = 50,
        max_f0: float = 800,
        delta_pitch: float = 0.005,
        penalty_factor: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_type = feature_type

        self.uses_cmvn = uses_cmvn
        self.uses_deltas = uses_deltas
        self.uses_splices = uses_splices
        self.uses_voiced = uses_voiced
        self.uses_speaker_adaptation = uses_speaker_adaptation

        self.frame_shift = frame_shift
        self.export_frame_shift = round(frame_shift / 1000, 4)
        self.frame_length = frame_length
        self.snip_edges = snip_edges

        # MFCC options

        self.use_energy = use_energy
        self.low_frequency = low_frequency
        self.high_frequency = high_frequency
        self.sample_frequency = sample_frequency
        self.allow_downsample = allow_downsample
        self.allow_upsample = allow_upsample
        self.dither = dither
        self.energy_floor = energy_floor
        self.num_coefficients = num_coefficients
        self.num_mel_bins = num_mel_bins
        self.cepstral_lifter = cepstral_lifter
        self.preemphasis_coefficient = preemphasis_coefficient

        # fMLLR options
        self.fmllr_update_type = fmllr_update_type
        self.silence_weight = silence_weight

        # Splicing options

        self.splice_left_context = splice_left_context
        self.splice_right_context = splice_right_context

        # Pitch features
        self.adaptive_pitch_range = adaptive_pitch_range
        self.use_pitch = use_pitch
        self.use_voicing = use_voicing
        self.use_delta_pitch = use_delta_pitch
        self.min_f0 = min_f0
        self.max_f0 = max_f0
        self.delta_pitch = delta_pitch
        self.penalty_factor = penalty_factor
        self.normalize_pitch = True
        if self.adaptive_pitch_range:
            self.min_f0 = 50
            self.max_f0 = 1200

    @property
    def vad_options(self) -> MetaDict:
        """Abstract method for VAD options"""
        raise NotImplementedError

    @property
    def alignment_model_path(self) -> str:  # needed for fmllr
        """Abstract method for alignment model path"""
        raise NotImplementedError

    @property
    def model_path(self) -> str:  # needed for fmllr
        """Abstract method for model path"""
        raise NotImplementedError

    @property
    @abstractmethod
    def working_directory(self) -> str:
        """Abstract method for working directory"""
        ...

    @property
    @abstractmethod
    def corpus_output_directory(self) -> str:
        """Abstract method for working directory of corpus"""
        ...

    @property
    @abstractmethod
    def data_directory(self) -> str:
        """Abstract method for corpus data directory"""
        ...

    @property
    def feature_options(self) -> MetaDict:
        """Parameters for feature generation"""
        options = {
            "type": self.feature_type,
            "use_energy": self.use_energy,
            "frame_shift": self.frame_shift,
            "frame_length": self.frame_length,
            "snip_edges": self.snip_edges,
            "low_frequency": self.low_frequency,
            "high_frequency": self.high_frequency,
            "sample_frequency": self.sample_frequency,
            "allow_downsample": self.allow_downsample,
            "allow_upsample": self.allow_upsample,
            "dither": self.dither,
            "energy_floor": self.energy_floor,
            "num_coefficients": self.num_coefficients,
            "num_mel_bins": self.num_mel_bins,
            "cepstral_lifter": self.cepstral_lifter,
            "preemphasis_coefficient": self.preemphasis_coefficient,
            "uses_cmvn": self.uses_cmvn,
            "uses_deltas": self.uses_deltas,
            "uses_voiced": self.uses_voiced,
            "uses_splices": self.uses_splices,
            "uses_speaker_adaptation": self.uses_speaker_adaptation,
            "use_pitch": self.use_pitch,
            "use_voicing": self.use_voicing,
            "min_f0": self.min_f0,
            "max_f0": self.max_f0,
            "delta_pitch": self.delta_pitch,
            "penalty_factor": self.penalty_factor,
            "silence_weight": self.silence_weight,
            "splice_left_context": self.splice_left_context,
            "splice_right_context": self.splice_right_context,
        }
        return options

    @abstractmethod
    def calc_fmllr(self) -> None:
        """Abstract method for calculating fMLLR transforms"""
        ...

    @property
    def fmllr_options(self) -> MetaDict:
        """Options for use in calculating fMLLR transforms"""
        return {
            "fmllr_update_type": self.fmllr_update_type,
            "silence_weight": self.silence_weight,
            "silence_csl": getattr(
                self, "silence_csl", ""
            ),  # If we have silence phones from a dictionary, use them
        }

    @property
    def lda_options(self) -> MetaDict:
        """Options for computing LDA"""
        return {
            "splice_left_context": self.splice_left_context,
            "splice_right_context": self.splice_right_context,
        }

    @property
    def mfcc_options(self) -> MetaDict:
        """Parameters to use in computing MFCC features."""
        return {
            "use-energy": self.use_energy,
            "dither": self.dither,
            "energy-floor": self.energy_floor,
            "num-ceps": self.num_coefficients,
            "num-mel-bins": self.num_mel_bins,
            "cepstral-lifter": self.cepstral_lifter,
            "preemphasis-coefficient": self.preemphasis_coefficient,
            "frame-shift": self.frame_shift,
            "frame-length": self.frame_length,
            "low-freq": self.low_frequency,
            "high-freq": self.high_frequency,
            "sample-frequency": self.sample_frequency,
            "allow-downsample": self.allow_downsample,
            "allow-upsample": self.allow_upsample,
            "snip-edges": self.snip_edges,
        }

    @property
    def pitch_options(self) -> MetaDict:
        """Parameters to use in computing MFCC features."""
        return {
            "use-pitch": self.use_pitch,
            "use-voicing": self.use_voicing,
            "use-delta-pitch": self.use_delta_pitch,
            "frame-shift": self.frame_shift,
            "frame-length": self.frame_length,
            "min-f0": self.min_f0,
            "max-f0": self.max_f0,
            "sample-frequency": self.sample_frequency,
            "penalty-factor": self.penalty_factor,
            "delta-pitch": self.delta_pitch,
            "snip-edges": self.snip_edges,
            "normalize": self.normalize_pitch,
        }


class VadConfigMixin(FeatureConfigMixin):
    """
    Abstract mixin class for performing voice activity detection

    Parameters
    ----------
    use_energy: bool
        Flag for using the first coefficient of MFCCs
    energy_threshold: float
        Energy threshold above which a frame will be counted as voiced
    energy_mean_scale: float
        Proportion of the mean energy of the file that should be added to the energy_threshold

    See Also
    --------
    :class:`~montreal_forced_aligner.corpus.features.FeatureConfigMixin`
        For feature generation parameters
    """

    def __init__(self, energy_threshold=5.5, energy_mean_scale=0.5, **kwargs):
        super().__init__(**kwargs)
        self.energy_threshold = energy_threshold
        self.energy_mean_scale = energy_mean_scale

    @property
    def vad_options(self) -> MetaDict:
        """Options for performing VAD"""
        return {
            "energy_threshold": self.energy_threshold,
            "energy_mean_scale": self.energy_mean_scale,
        }


class IvectorConfigMixin(VadConfigMixin):
    """
    Mixin class for ivector features

    Parameters
    ----------
    ivector_dimension: int
        Dimension of ivectors
    num_gselect: int
        Gaussian-selection using diagonal model: number of Gaussians to select
    posterior_scale: float
        Scale on the acoustic posteriors, intended to account for inter-frame correlations
    min_post : float
        Minimum posterior to use (posteriors below this are pruned out)
    max_count: int
        The use of this option (e.g. --max-count 100) can make iVectors more consistent for different lengths of
        utterance, by scaling up the prior term when the data-count exceeds this value. The data-count is after
        posterior-scaling, so assuming the posterior-scale is 0.1, --max-count 100 starts having effect after 1000
        frames, or 10 seconds of data.

    See Also
    --------
    :class:`~montreal_forced_aligner.corpus.features.FeatureConfigMixin`
        For feature generation parameters
    """

    def __init__(
        self,
        num_gselect: int = 20,
        posterior_scale: float = 1.0,
        min_post: float = 0.025,
        max_count: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ivector_dimension = IVECTOR_DIMENSION
        self.num_gselect = num_gselect
        self.posterior_scale = posterior_scale
        self.min_post = min_post
        self.max_count = max_count
        self.normalize_pitch = False

    @abstractmethod
    def extract_ivectors(self) -> None:
        """Abstract method for extracting ivectors"""
        ...

    @property
    def ivector_options(self) -> MetaDict:
        """Options for ivector training and extracting"""
        return {
            "num_gselect": self.num_gselect,
            "posterior_scale": self.posterior_scale,
            "min_post": self.min_post,
            "silence_weight": self.silence_weight,
            "max_count": self.max_count,
            "ivector_dimension": self.ivector_dimension,
            "silence_csl": getattr(
                self, "silence_csl", ""
            ),  # If we have silence phones from a dictionary, use them,
        }


class ExtractIvectorsFunction(KaldiFunction):
    """
    Multiprocessing function for extracting ivectors.

    See Also
    --------
    :meth:`.IvectorCorpusMixin.extract_ivectors`
        Main function that calls this function in parallel
    :meth:`.IvectorCorpusMixin.extract_ivectors_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`ivector-extract`
        Relevant Kaldi binary
    :kaldi_src:`gmm-global-get-post`
        Relevant Kaldi binary
    :kaldi_src:`weight-silence-post`
        Relevant Kaldi binary
    :kaldi_src:`weight-post`
        Relevant Kaldi binary
    :kaldi_src:`post-to-weights`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.corpus.features.ExtractIvectorsArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(r"^VLOG.*Ivector norm for utterance (?P<utterance>.+) was.*")

    def __init__(self, args: ExtractIvectorsArguments):
        super().__init__(args)
        self.ivector_options = args.ivector_options
        self.ie_path = args.ie_path
        self.ivectors_scp_path = args.ivectors_scp_path
        self.dubm_path = args.dubm_path

    def _run(self) -> typing.Generator[str]:
        """Run the function"""
        if os.path.exists(self.ivectors_scp_path):
            return
        with Session(self.db_engine) as session, mfa_open(self.log_path, "w") as log_file:
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True))
                .filter(Job.id == self.job_name)
                .first()
            )
            feature_string = job.construct_online_feature_proc_string()

            gmm_global_get_post_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-global-get-post"),
                    f"--n={self.ivector_options['num_gselect']}",
                    f"--min-post={self.ivector_options['min_post']}",
                    self.dubm_path,
                    feature_string,
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            ivector_ark_path = self.ivectors_scp_path.replace(".scp", ".ark")
            extract_proc = subprocess.Popen(
                [
                    thirdparty_binary("ivector-extract"),
                    "--verbose=2",
                    f"--acoustic-weight={self.ivector_options['posterior_scale']}",
                    "--compute-objf-change=true",
                    f"--max-count={self.ivector_options['max_count']}",
                    self.ie_path,
                    feature_string,
                    "ark,s,cs:-",
                    f"ark,scp:{ivector_ark_path},{self.ivectors_scp_path}",
                ],
                stderr=subprocess.PIPE,
                encoding="utf8",
                stdin=gmm_global_get_post_proc.stdout,
                env=os.environ,
            )
            for line in extract_proc.stderr:
                log_file.write(line)
                log_file.flush()
                m = self.progress_pattern.match(line.strip())
                if m:
                    yield m.group("utterance")


@njit
def plda_distance(train_ivector: np.ndarray, test_ivector: np.ndarray, psi: np.ndarray):
    """
    Distance formulation of PLDA log likelihoods. Positive log likelihood ratios are transformed
    into 1 / log likelihood ratio and negative log likelihood ratios are made positive.

    Parameters
    ----------
    train_ivector: numpy.ndarray
        Utterance ivector to use as reference
    test_ivector: numpy.ndarray
        Utterance ivector to compare
    psi: numpy.ndarray
        Input psi from :class:`~montreal_forced_aligner.corpus.features.PldaModel`

    Returns
    -------
    float
        PLDA distance
    """
    max_log_likelihood = 40.0
    loglike = plda_log_likelihood(train_ivector, test_ivector, psi)
    if loglike >= max_log_likelihood:
        return 0.0
    return max_log_likelihood - loglike


@njit(cache=True)
def plda_variance_given(psi: np.ndarray, train_count: int = None):
    if train_count is not None:
        variance_given = 1.0 + psi / (train_count * psi + 1.0)
    else:
        variance_given = 1.0 + psi / (psi + 1.0)
    logdet_given = np.sum(np.log(variance_given))
    variance_given = 1.0 / variance_given
    return logdet_given, variance_given


@njit(cache=True)
def plda_variance_without(psi: np.ndarray):
    variance_without = 1.0 + psi
    logdet_without = np.sum(np.log(variance_without))
    variance_without = 1.0 / variance_without
    return logdet_without, variance_without


@njit
def plda_log_likelihood(
    train_ivector: np.ndarray, test_ivector: np.ndarray, psi: np.ndarray, train_count: int = None
):
    """
    Calculate log likelihood of two ivectors belonging to the same class

    Parameters
    ----------
    train_ivector: numpy.ndarray
        Speaker or utterance ivector to use as reference
    test_ivector: numpy.ndarray
        Utterance ivector to compare
    psi: numpy.ndarray
        Input psi from :class:`~montreal_forced_aligner.corpus.features.PldaModel`
    train_count: int, optional
        Count of training ivector, if it represents a speaker

    Returns
    -------
    float
        Log likelihood ratio of same class hypothesis compared to difference class hypothesis
    """
    train_ivector = train_ivector.astype("float64")
    test_ivector = test_ivector.astype("float64")
    psi = psi.astype("float64")
    if train_count is not None:
        mean = (train_count * psi) / (train_count * psi + 1.0)
        mean *= train_ivector  # N X D , X[0]- Train ivectors
    else:
        mean = (psi) / (psi + 1.0)
        mean *= train_ivector  # N X D , X[0]- Train ivectors
    logdet_given, variance_given = plda_variance_given(psi, train_count)
    # without class computation
    logdet_without, variance_without = plda_variance_without(psi)
    sqdiff_given = test_ivector - mean
    sqdiff_given = sqdiff_given**2
    loglikes = -0.5 * (
        logdet_given + M_LOG_2PI * PLDA_DIMENSION + np.dot(sqdiff_given, variance_given)
    )
    sqdiff_without = test_ivector**2
    loglike_without_class = -0.5 * (
        logdet_without + M_LOG_2PI * PLDA_DIMENSION + np.dot(sqdiff_without, variance_without)
    )
    return loglikes - loglike_without_class


@njit(parallel=True)
def plda_distance_matrix(
    train_ivectors: np.ndarray,
    test_ivectors: np.ndarray,
    psi: np.ndarray,
) -> np.ndarray:
    """
    Adapted from https://github.com/prachiisc/PLDA_scoring/blob/master/PLDA_scoring.py#L177
    Computes plda affinity matrix using Loglikelihood function

    Parameters
    ----------
    train_ivectors : numpy.ndarray
        Ivectors to compare test ivectors against against 1 X N X D
    test_ivectors : numpy.ndarray
        Ivectors to compare against training examples 1 X M X D
    normalize: bool
        Flag for normalizing matrix by the maximum value
    distance: bool
        Flag for converting PLDA log likelihood ratios into a distance metric

    Returns
    -------
    np.ndarray
        Affinity matrix, shape is number of train ivectors by the number of test ivectors (M X N)
    """
    num_train = train_ivectors.shape[0]
    num_test = test_ivectors.shape[0]
    distance_matrix = np.zeros((num_test, num_train))
    for i in numba.prange(num_train):
        for j in numba.prange(num_test):
            distance_matrix[i, j] = plda_log_likelihood(train_ivectors[i], test_ivectors[j], psi)
    return distance_matrix


def pairwise_plda_distance_matrix(
    ivectors: np.ndarray,
    psi: np.ndarray,
) -> csr_matrix:
    """
    Adapted from https://github.com/prachiisc/PLDA_scoring/blob/master/PLDA_scoring.py#L177
    Computes plda affinity matrix using Loglikelihood function

    Parameters
    ----------
    train_ivectors : numpy.ndarray
        Ivectors to compare test ivectors against against 1 X N X D
    test_ivectors : numpy.ndarray
        Ivectors to compare against training examples 1 X M X D
    normalize: bool
        Flag for normalizing matrix by the maximum value
    distance: bool
        Flag for converting PLDA log likelihood ratios into a distance metric

    Returns
    -------
    np.ndarray
        Affinity matrix, shape is number of train ivectors by the number of test ivectors (M X N)
    """
    full = plda_distance_matrix(ivectors, ivectors, psi)
    return csr_matrix(full[np.where(full > 5)])


@njit(parallel=True)
def score_plda(
    train_ivectors: np.ndarray,
    test_ivectors: np.ndarray,
    psi: np.ndarray,
    normalize=False,
    distance=False,
) -> np.ndarray:
    """
    Adapted from https://github.com/prachiisc/PLDA_scoring/blob/master/PLDA_scoring.py#L177
    Computes plda affinity matrix using Loglikelihood function

    Parameters
    ----------
    train_ivectors : numpy.ndarray
        Ivectors to compare test ivectors against against 1 X N X D
    test_ivectors : numpy.ndarray
        Ivectors to compare against training examples 1 X M X D
    normalize: bool
        Flag for normalizing matrix by the maximum value
    distance: bool
        Flag for converting PLDA log likelihood ratios into a distance metric

    Returns
    -------
    np.ndarray
        Affinity matrix, shape is number of train ivectors by the number of test ivectors (M X N)
    """
    mean = (psi) / (psi + 1.0)
    mean = mean.reshape(1, -1) * train_ivectors

    # given class computation
    variance_given = 1.0 + psi / (psi + 1.0)
    logdet_given = np.sum(np.log(variance_given))
    variance_given = 1.0 / variance_given

    # without class computation
    variance_without = 1.0 + psi
    logdet_without = np.sum(np.log(variance_without))
    variance_without = 1.0 / variance_without

    sqdiff = test_ivectors  # ---- Test x-vectors
    num_train = train_ivectors.shape[0]
    num_test = test_ivectors.shape[0]
    dim = test_ivectors.shape[1]
    loglikes = np.zeros((num_test, num_train))
    sqdiff_without = sqdiff**2
    loglike_without_class = -0.5 * (
        logdet_without + M_LOG_2PI * dim + (sqdiff_without @ variance_without)
    )
    for i in numba.prange(num_train):
        sqdiff_given = sqdiff - mean[i]
        sqdiff_given = sqdiff_given**2
        loglikes[:, i] = (
            -0.5 * (logdet_given + M_LOG_2PI * dim + (sqdiff_given @ variance_given))
        ) - loglike_without_class

    if distance:
        threshold = np.max(loglikes)
        loglikes -= threshold
        loglikes *= -1
    if normalize:
        # loglike_ratio -= np.min(loglike_ratio)
        loglikes /= threshold
    return loglikes


@njit
def compute_classification_stats(
    speaker_ivectors: np.ndarray, psi: np.ndarray, counts: np.ndarray
):
    mean = (counts.reshape(-1, 1) * psi.reshape(1, -1)) / (
        counts.reshape(-1, 1) * psi.reshape(1, -1) + 1.0
    )
    mean = mean * speaker_ivectors  # N X D , X[0]- Train ivectors
    # given class computation
    variance_given = 1.0 + psi / (counts.reshape(-1, 1) * psi.reshape(1, -1) + 1.0)
    logdet_given = np.sum(np.log(variance_given), axis=1)
    variance_given = 1.0 / variance_given

    # without class computation
    variance_without = 1.0 + psi
    logdet_without = np.sum(np.log(variance_without))
    variance_without = 1.0 / variance_without
    return mean, variance_given, logdet_given, variance_without, logdet_without


@njit(parallel=True)
def classify_plda(
    utterance_ivector: np.ndarray,
    mean,
    variance_given,
    logdet_given,
    variance_without,
    logdet_without,
) -> typing.Tuple[int, float]:
    """
    Adapted from https://github.com/prachiisc/PLDA_scoring/blob/master/PLDA_scoring.py#L177
    Computes plda affinity matrix using Loglikelihood function

    Parameters
    ----------
    utterance_ivector : numpy.ndarray
        Utterance ivector to compare against

    Returns
    -------
    int
        Best speaker index
    float
        Best speaker PLDA score
    """

    num_speakers = mean.shape[0]

    sqdiff_without = utterance_ivector**2
    loglike_without_class = -0.5 * (
        logdet_without + M_LOG_2PI * PLDA_DIMENSION + (sqdiff_without @ variance_without)
    )
    loglikes = np.zeros((num_speakers,))
    for i in numba.prange(num_speakers):
        sqdiff_given = utterance_ivector - mean[i]
        sqdiff_given = sqdiff_given**2
        logdet = logdet_given[i]
        variance = variance_given[i]

        loglikes[i] = (
            -0.5 * (logdet + M_LOG_2PI * PLDA_DIMENSION + (sqdiff_given @ variance))
        ) - loglike_without_class

    ind = loglikes.argmax()
    return ind, loglikes[ind]


@njit(parallel=True)
def score_plda_train_counts(
    train_ivectors: np.ndarray, test_ivectors: np.ndarray, psi: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """
    Adapted from https://github.com/prachiisc/PLDA_scoring/blob/master/PLDA_scoring.py#L177
    Computes plda affinity matrix using Loglikelihood function

    Parameters
    ----------
    train_ivectors : numpy.ndarray
        Ivectors to compare test ivectors against against 1 X N X D
    test_ivectors : numpy.ndarray
        Ivectors to compare against training examples 1 X M X D
    normalize: bool
        Flag for normalizing matrix by the maximum value
    distance: bool
        Flag for converting PLDA log likelihood ratios into a distance metric

    Returns
    -------
    np.ndarray
        Affinity matrix, shape is number of train ivectors by the number of test ivectors (M X N)
    """
    num_train = train_ivectors.shape[0]
    num_test = test_ivectors.shape[0]
    loglikes = np.zeros((num_test, num_train))
    for i in numba.prange(num_train):
        for j in numba.prange(num_test):
            loglikes[j, i] = plda_log_likelihood(
                train_ivectors[i], test_ivectors[j], psi, counts[i]
            )
    return loglikes


@dataclassy.dataclass(slots=True)
class PldaModel:
    """PLDA model for transforming and scoring ivectors based on log likelihood ratios"""

    mean: np.ndarray
    diagonalizing_transform: np.ndarray
    psi: np.ndarray
    offset: typing.Optional[np.ndarray] = None
    pca_transform: typing.Optional[np.ndarray] = None
    transformed_mean: typing.Optional[np.ndarray] = None
    transformed_diagonalizing_transform: typing.Optional[np.ndarray] = None

    @classmethod
    def load(cls, plda_path):
        """
        Instantiate a PLDA model from a trained model file

        Parameters
        ----------
        plda_path: str
            Path to trained PLDA model

        Returns
        -------
        :class:`~montreal_forced_aligner.corpus.features.PldaModel`
            Instantiated object
        """
        mean = None
        diagonalizing_transform = None
        diagonalizing_transform_lines = []
        psi = None
        copy_proc = subprocess.Popen(
            [thirdparty_binary("ivector-copy-plda"), "--binary=false", plda_path, "-"],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            env=os.environ,
            encoding="utf8",
        )
        for line in copy_proc.stdout:
            if mean is None:
                line = line.replace("<Plda>", "").strip()[2:-2]
                mean = np.fromstring(line, sep=" ")
            elif diagonalizing_transform is None:
                if "[" in line:
                    continue
                end_mat = "]" in line
                line = line.replace("[", "").replace("]", "").strip()
                row = np.fromstring(line, sep=" ")
                diagonalizing_transform_lines.append(row)
                if end_mat:
                    diagonalizing_transform = np.array(diagonalizing_transform_lines)
            elif psi is None:
                line = line.strip()[2:-2]
                psi = np.fromstring(line, sep=" ")
        copy_proc.wait()
        offset = -diagonalizing_transform @ mean.reshape(-1, 1)
        return PldaModel(mean, diagonalizing_transform, psi, offset)

    def distance(self, train_ivector: np.ndarray, test_ivector: np.ndarray):
        """
        Distance formulation of PLDA log likelihoods. Positive log likelihood ratios are transformed
        into 1 / log likelihood ratio and negative log likelihood ratios are made positive.

        Parameters
        ----------
        train_ivector: numpy.ndarray
            Utterance ivector to use as reference
        test_ivector: numpy.ndarray
            Utterance ivector to compare

        Returns
        -------
        float
            PLDA distance
        """
        return plda_distance(train_ivector, test_ivector, self.psi)

    def log_likelihood(self, train_ivector: np.ndarray, test_ivector: np.ndarray, count: int = 1):
        """
        Calculate log likelihood of two ivectors belonging to the same class

        Parameters
        ----------
        train_ivector: numpy.ndarray
            Speaker or utterance ivector to use as reference
        test_ivector: numpy.ndarray
            Utterance ivector to compare
        count: int, optional
            Count of training ivector, if it represents a speaker

        Returns
        -------
        float
            Log likelihood ratio of same class hypothesis compared to difference class hypothesis
        """
        return plda_log_likelihood(train_ivector, test_ivector, self.psi, count)

    def process_ivectors(self, ivectors: np.ndarray, counts: np.ndarray = None) -> np.ndarray:
        """
        Transform ivectors to PLDA space

        Parameters
        ----------
        ivectors: numpy.ndarray
            Ivectors to process
        counts: numpy.ndarray, optional
            Number of utterances if ivectors are per-speaker

        Returns
        -------
        numpy.ndarray
            Transformed ivectors
        """
        # ivectors = self.preprocess_ivectors(ivectors)
        # ivectors = self.compute_pca_transform(ivectors)
        ivectors = self.transform_ivectors(ivectors, counts=counts)
        return ivectors

    def preprocess_ivectors(self, ivectors: np.ndarray) -> np.ndarray:
        """
        Adapted from https://github.com/prachiisc/PLDA_scoring/blob/master/PLDA_scoring.py#L25

        Parameters
        ----------
        ivectors: numpy.ndarray
            Input ivectors

        Returns
        -------
        numpy.ndarray
            Preprocessed ivectors
        """
        ivectors = ivectors.T  # DX N
        dim = ivectors.shape[1]
        # preprocessing
        # mean subtraction
        ivectors = ivectors - self.mean[:, np.newaxis]
        # PCA transform
        # ivectors = self.diagonalizing_transform @ ivectors
        l2_norm = np.linalg.norm(ivectors, axis=0, keepdims=True)
        l2_norm = l2_norm / math.sqrt(dim)

        ivectors_new = ivectors / l2_norm

        return ivectors_new.T

    def compute_pca_transform(self, ivectors: np.ndarray) -> np.ndarray:
        """
        Adapted from https://github.com/prachiisc/PLDA_scoring/blob/master/PLDA_scoring.py#L53

        Apply transform on mean shifted ivectors

        Parameters
        ----------
        ivectors: numpy.ndarray
            Input ivectors

        Returns
        ----------
        numpy.ndarray
            Transformed ivectors
        """
        if PLDA_DIMENSION == IVECTOR_DIMENSION:
            return ivectors
        if self.pca_transform is not None:
            return ivectors @ self.pca_transform
        num_rows = ivectors.shape[0]
        mean = np.mean(ivectors, 0, keepdims=True)
        S = np.matmul(ivectors.T, ivectors)
        S = S / num_rows

        S = S - mean.T @ mean

        ev_s, eig_s, _ = np.linalg.svd(S, full_matrices=True)
        energy_percent = np.sum(eig_s[:PLDA_DIMENSION]) / np.sum(eig_s)
        logger.debug(f"PLDA PCA transform energy with: {energy_percent*100:.2f}%")
        transform = ev_s[:, :PLDA_DIMENSION]

        transxvec = ivectors @ transform
        newX = transxvec
        self.pca_transform = transform
        self.apply_transform()
        return newX

    def apply_transform(self):
        """
        Adapted from https://github.com/prachiisc/PLDA_scoring/blob/master/PLDA_scoring.py#L101

        Parameters
        ----------
        transform_in : numpy.ndarray
           PCA transform
        """

        mean_plda = self.mean
        # transfomed mean vector
        transform_in = self.pca_transform.T
        new_mean = transform_in @ mean_plda[:, np.newaxis]
        D = self.diagonalizing_transform
        psi = self.psi
        D_inv = np.linalg.inv(D)
        # within class and between class covarinace
        phi_b = (D_inv * psi.reshape(1, -1)) @ D_inv.T
        phi_w = D_inv @ D_inv.T
        # transformed with class and between class covariance
        new_phi_b = transform_in @ phi_b @ transform_in.T
        new_phi_w = transform_in @ phi_w @ transform_in.T
        ev_w, eig_w, _ = np.linalg.svd(new_phi_w)
        eig_w_inv = 1 / np.sqrt(eig_w)
        Dnew = eig_w_inv.reshape(-1, 1) * ev_w.T
        new_phi_b_proj = Dnew @ new_phi_b @ Dnew.T
        ev_b, eig_b, _ = np.linalg.svd(new_phi_b_proj)
        psi_new = eig_b

        Dnew = ev_b.T @ Dnew
        self.transformed_mean = new_mean
        self.transformed_diagonalizing_transform = Dnew
        self.psi = psi_new
        self.offset = -Dnew @ new_mean.reshape(-1, 1)

    def transform_ivectors(self, ivectors: np.ndarray, counts: np.ndarray = None) -> np.ndarray:
        """
        Adapted from https://github.com/prachiisc/PLDA_scoring/blob/master/PLDA_scoring.py#L142
        Apply plda mean and diagonalizing transform to ivectors for scoring

        Parameters
        ----------
        ivectors : numpy.ndarray
           Input ivectors

        Returns
        -------
        numpy.ndarray
            transformed ivectors
        """

        offset = self.offset
        offset = offset.T
        if PLDA_DIMENSION == IVECTOR_DIMENSION:
            D = self.diagonalizing_transform
        else:
            D = self.transformed_diagonalizing_transform
        Dnew = D.T
        X_new = ivectors @ Dnew
        X_new = X_new + offset
        # Get normalizing factor
        # Defaults : normalize_length(true), simple_length_norm(false)
        X_new_sq = X_new**2

        if counts is not None:
            dot_prod = np.zeros((X_new.shape[0], 1))
            for i in range(dot_prod.shape[0]):
                inv_covar = self.psi + (1.0 / counts[i])
                inv_covar = 1.0 / inv_covar
                dot_prod[i] = np.dot(X_new_sq[i], inv_covar)
        else:
            inv_covar = (1.0 / (1.0 + self.psi)).reshape(-1, 1)
            dot_prod = X_new_sq @ inv_covar  # N X 1
        Dim = D.shape[0]
        normfactor = np.sqrt(Dim / dot_prod)

        X_new = X_new * normfactor

        return X_new


class ExportIvectorsFunction(KaldiFunction):
    """
    Multiprocessing function to compute voice activity detection
    See Also
    --------
    :meth:`.AcousticCorpusMixin.compute_vad`
        Main function that calls this function in parallel
    :meth:`.AcousticCorpusMixin.compute_vad_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`compute-vad`
        Relevant Kaldi binary
    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.corpus.features.VadArguments`
        Arguments for the function
    """

    def __init__(self, args: ExportIvectorsArguments):
        super().__init__(args)
        self.use_xvector = args.use_xvector

    def _run(self) -> typing.Generator[typing.Tuple[int, int, int]]:
        """Run the function"""
        engine = sqlalchemy.create_engine(
            self.db_string,
            poolclass=sqlalchemy.NullPool,
            pool_reset_on_return=None,
            isolation_level="AUTOCOMMIT",
            logging_name=f"{type(self).__name__}_engine",
        ).execution_options(logging_token=f"{type(self).__name__}_engine")
        with sqlalchemy.orm.Session(engine) as session, mfa_open(self.log_path, "w") as log_file:

            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True))
                .filter(Job.id == self.job_name)
                .first()
            )
            if self.use_xvector:
                ivector_column = Utterance.xvector
            else:
                ivector_column = Utterance.ivector
            query = (
                session.query(Utterance.kaldi_id, ivector_column)
                .filter(ivector_column != None, Utterance.job_id == job.id)  # noqa
                .order_by(Utterance.kaldi_id)
            )

            ivector_scp_path = job.construct_path(job.corpus.split_directory, "ivectors", "scp")
            ivector_ark_path = job.construct_path(job.corpus.split_directory, "ivectors", "ark")
            input_proc = subprocess.Popen(
                [
                    thirdparty_binary("copy-vector"),
                    "--binary=true",
                    "ark,t:-",
                    f"ark,scp:{ivector_ark_path},{ivector_scp_path}",
                ],
                stdin=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            for utt_id, ivector in query:
                if ivector is None:
                    continue
                ivector = " ".join([format(x, ".12g") for x in ivector])
                in_line = f"{utt_id}  [ {ivector} ]\n".encode("utf8")
                input_proc.stdin.write(in_line)
                input_proc.stdin.flush()
            input_proc.stdin.close()
            self.check_call(input_proc)
            with mfa_open(ivector_scp_path) as f:
                for line in f:
                    line = line.strip()
                    utt_id, ark_path = line.split(maxsplit=1)
                    utt_id = int(utt_id.split("-")[1])
                    yield utt_id, ark_path
