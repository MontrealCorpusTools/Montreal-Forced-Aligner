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
import numpy as np
import sqlalchemy
from sqlalchemy.orm import Session, joinedload

from montreal_forced_aligner.abc import KaldiFunction
from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.data import MfaArguments
from montreal_forced_aligner.db import CorpusWorkflow, Job, Utterance
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.utils import thirdparty_binary

if TYPE_CHECKING:
    SpeakerCharacterType = Union[str, int]
    from montreal_forced_aligner.abc import MetaDict

__all__ = [
    "FeatureConfigMixin",
    "CalcFmllrFunction",
    "ComputeVadFunction",
    "VadArguments",
    "MfccArguments",
    "CalcFmllrArguments",
    "ExtractIvectorsFunction",
    "ExtractIvectorsArguments",
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
class RecomputeVadArguments(MfaArguments):
    """Arguments for :class:`~montreal_forced_aligner.corpus.features.ComputeVadFunction`"""

    dubm_model: str
    vad_options: MetaDict
    ivector_options: MetaDict


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


def compute_feature_process(
    log_file: io.FileIO,
    wav_path: str,
    segment_path: str,
    mfcc_options: MetaDict,
    pitch_options: MetaDict,
    min_length=0.1,
    no_logging=False,
) -> typing.Tuple[typing.Optional[subprocess.Popen], subprocess.Popen]:
    """
    Construct processes for computing features

    Parameters
    ----------
    log_file: io.FileIO
        File for logging stderr
    wav_path: str
        Wav scp to use
    segment_path: str
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
        Feature pasting process
    subprocess.Popen
        Computation process for progress information
    """
    use_pitch = pitch_options.pop("use-pitch")
    use_voicing = pitch_options.pop("use-voicing")
    mfcc_base_command = [thirdparty_binary("compute-mfcc-feats")]
    for k, v in mfcc_options.items():
        mfcc_base_command.append(f"--{k.replace('_', '-')}={feature_make_safe(v)}")
    comp_proc_logger = subprocess.PIPE
    if no_logging:
        comp_proc_logger = log_file
    if os.path.exists(segment_path):
        mfcc_base_command += ["ark:-", "ark:-"]
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
        comp_proc = subprocess.Popen(
            mfcc_base_command,
            stdout=subprocess.PIPE,
            stderr=comp_proc_logger,
            stdin=seg_proc.stdout,
            env=os.environ,
        )
    else:
        mfcc_base_command += [f"scp,p:{wav_path}", "ark:-"]
        comp_proc = subprocess.Popen(
            mfcc_base_command,
            stdout=subprocess.PIPE,
            stderr=comp_proc_logger,
            env=os.environ,
        )
    if not use_pitch and not use_voicing:
        return None, comp_proc
    pitch_base_command = [
        thirdparty_binary("compute-and-process-kaldi-pitch-feats"),
    ]
    for k, v in pitch_options.items():
        pitch_base_command.append(f"--{k.replace('_', '-')}={feature_make_safe(v)}")
        if k == "delta-pitch":
            pitch_base_command.append(f"--delta-pitch-noise-stddev={feature_make_safe(v)}")
    if use_pitch:
        pitch_base_command.append("--add-delta-pitch=true")
        pitch_base_command.append("--add-normalized-log-pitch=true")
    else:
        pitch_base_command.append("--add-delta-pitch=false")
        pitch_base_command.append("--add-normalized-log-pitch=false")
    if use_voicing:
        pitch_base_command.append("--add-pov-feature=true")
    else:
        pitch_base_command.append("--add-pov-feature=false")
    pitch_command = " ".join(pitch_base_command)
    if os.path.exists(segment_path):
        segment_command = f'extract-segments --min-segment-length={min_length} scp:"{wav_path}" "{segment_path}" ark:- | '
        pitch_input = "ark:-"
    else:
        segment_command = ""
        pitch_input = f'scp:"{wav_path}"'
    pitch_feat_string = f"ark,s,cs:{segment_command}{pitch_command} {pitch_input} ark:- |"
    length_tolerance = 2
    paste_proc = subprocess.Popen(
        [
            thirdparty_binary("paste-feats"),
            f"--length-tolerance={length_tolerance}",
            "ark,s,cs:-",
            pitch_feat_string,
            "ark:-",
        ],
        stdin=comp_proc.stdout,
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=log_file,
    )
    return paste_proc, comp_proc


def compute_transform_process(
    log_file: io.FileIO,
    feat_proc: subprocess.Popen,
    utt2spk_path: str,
    cmvn_path: str,
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
    cmvn_proc = subprocess.Popen(
        ["apply-cmvn", f"--utt2spk=ark:{utt2spk_path}", f"scp:{cmvn_path}", "ark:-", "ark:-"],
        env=os.environ,
        stdin=feat_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=log_file,
    )
    if lda_mat_path is not None:
        splice_proc = subprocess.Popen(
            [
                "splice-feats",
                f'--left-context={lda_options["splice_left_context"]}',
                f'--right-context={lda_options["splice_right_context"]}',
                "ark,s,cs:-",
                "ark:-",
            ],
            env=os.environ,
            stdin=cmvn_proc.stdout,
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
            ["add-deltas", "ark,s,cs:-", "ark:-"],
            env=os.environ,
            stdin=cmvn_proc.stdout,
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
        self.mfcc_options = args.mfcc_options
        self.pitch_options = args.pitch_options

    def _run(self) -> typing.Generator[int]:
        """Run the function"""
        db_engine = sqlalchemy.create_engine(self.db_string)
        with Session(db_engine) as session, mfa_open(self.log_path, "w") as log_file:
            processed = 0
            job: Job = session.get(Job, self.job_name)
            feats_scp_path = job.construct_path(self.data_directory, "feats", "scp")
            wav_path = job.construct_path(self.data_directory, "wav", "scp")
            segment_path = job.construct_path(self.data_directory, "segments", "scp")
            raw_ark_path = job.construct_path(self.data_directory, "feats", "ark")
            if os.path.exists(raw_ark_path):
                return
            paste_proc, comp_proc = compute_feature_process(
                log_file, wav_path, segment_path, self.mfcc_options, self.pitch_options
            )
            copy_proc = subprocess.Popen(
                [
                    thirdparty_binary("copy-feats"),
                    "--verbose=2",
                    "--compress=true",
                    "ark,s,cs:-",
                    f"ark,scp:{raw_ark_path},{feats_scp_path}",
                ],
                stdin=paste_proc.stdout if paste_proc is not None else comp_proc.stdout,
                stderr=log_file,
                env=os.environ,
                encoding="utf8",
            )

            for line in comp_proc.stderr:
                line = line.strip().decode("utf8")
                log_file.write(line + "\n")
                m = self.progress_pattern.match(line)
                if m:
                    cur = int(m.group("num_utterances"))
                    increment = cur - processed
                    processed = cur
                    yield increment
            self.check_call(copy_proc)
        db_engine.dispose()


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


class RecomputeVadFunction(KaldiFunction):
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

    def __init__(self, args: RecomputeVadArguments):
        super().__init__(args)
        self.dubm_model = args.dubm_model
        self.vad_options = args.vad_options
        self.ivector_options = args.ivector_options

    def _run(self) -> typing.Generator[typing.Tuple[int, int, int]]:
        """Run the function"""
        db_engine = sqlalchemy.create_engine(self.db_string)
        with Session(db_engine) as session, mfa_open(self.log_path, "w") as log_file:
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True))
                .filter(Job.id == self.job_name)
                .first()
            )
            workflow: CorpusWorkflow = (
                session.query(CorpusWorkflow)
                .filter(CorpusWorkflow.current == True)  # noqa
                .first()
            )
            feature_string = job.construct_online_feature_proc_string(uses_vad=False)
            merge_map_path = os.path.join(workflow.working_directory, "merge_vad_map.txt")
            vad_scp_path = job.construct_path(job.corpus.current_subset_directory, "vad", "scp")
            vad_ark_path = job.construct_path(job.corpus.current_subset_directory, "vad", "ark")
            merged_vad_scp_path = job.construct_path(
                job.corpus.current_subset_directory, "merged_vad", "scp"
            )
            merged_vad_ark_path = job.construct_path(
                job.corpus.current_subset_directory, "merged_vad", "ark"
            )

            gselect_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-gselect"),
                    f"--n={self.ivector_options['num_gselect']}",
                    self.dubm_model,
                    feature_string,
                    "ark:-",
                ],
                stderr=log_file,
                stdout=subprocess.PIPE,
                env=os.environ,
            )

            frame_like_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-global-get-frame-likes"),
                    "--average=false",
                    "--gselect=ark,s,cs:-",
                    self.dubm_model,
                    feature_string,
                    "ark:-",
                ],
                stderr=log_file,
                stdout=subprocess.PIPE,
                stdin=gselect_proc.stdout,
                env=os.environ,
            )
            gmm_vad_proc = subprocess.Popen(
                [
                    thirdparty_binary("compute-vad-from-frame-likes"),
                    "ark,s,cs:-",
                    "ark:-",
                ],
                stderr=log_file,
                stdout=subprocess.PIPE,
                stdin=frame_like_proc.stdout,
                env=os.environ,
            )
            merge_vad_proc = subprocess.Popen(
                [
                    thirdparty_binary("merge-vads"),
                    f"--map={merge_map_path}" f"scp,s,cs:{vad_scp_path}",
                    "ark,s,cs:-",
                    f"ark,scp:{merged_vad_ark_path},{merged_vad_scp_path}",
                ],
                stderr=log_file,
                stdin=gmm_vad_proc.stdout,
                env=os.environ,
            )
            for line in merge_vad_proc.stderr:
                log_file.write(line)
                yield 1
                # m = self.progress_pattern.match(line.strip())
                # if m:
                #    yield int(m.group("done")), int(m.group("no_feats")), int(m.group("unvoiced"))
            self.check_call(merge_vad_proc)
            os.remove(vad_scp_path)
            os.remove(vad_ark_path)
            os.rename(merged_vad_scp_path, vad_scp_path)
            os.rename(merged_vad_ark_path, vad_ark_path)
        db_engine.dispose()


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
        uses_speaker_adaptation: bool = False,
        fmllr_update_type: str = "full",
        silence_weight: float = 0.0,
        splice_left_context: int = 3,
        splice_right_context: int = 3,
        use_pitch: bool = False,
        use_voicing: bool = False,
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
        self.use_pitch = use_pitch
        self.use_voicing = use_voicing
        self.min_f0 = min_f0
        self.max_f0 = max_f0
        self.delta_pitch = delta_pitch
        self.penalty_factor = penalty_factor

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
            "frame-shift": self.frame_shift,
            "frame-length": self.frame_length,
            "min-f0": self.min_f0,
            "max-f0": self.max_f0,
            "sample-frequency": self.sample_frequency,
            "penalty-factor": self.penalty_factor,
            "delta-pitch": self.delta_pitch,
            "snip-edges": self.snip_edges,
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
        ivector_dimension=128,
        num_gselect=20,
        posterior_scale=1.0,
        min_post=0.025,
        max_count=100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ivector_dimension = ivector_dimension
        self.num_gselect = num_gselect
        self.posterior_scale = posterior_scale
        self.min_post = min_post
        self.max_count = max_count

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
        db_engine = sqlalchemy.create_engine(self.db_string)
        with Session(db_engine) as session, mfa_open(self.log_path, "w") as log_file:
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True))
                .filter(Job.id == self.job_name)
                .first()
            )
            feature_string = job.construct_online_feature_proc_string(uses_vad=True)

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
        db_engine.dispose()


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

    def __init__(self, args: MfaArguments):
        super().__init__(args)

    def _run(self) -> typing.Generator[typing.Tuple[int, int, int]]:
        """Run the function"""
        engine = sqlalchemy.create_engine(self.db_string)
        with sqlalchemy.orm.Session(engine) as session, mfa_open(self.log_path, "w") as log_file:

            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True))
                .filter(Job.id == self.job_name)
                .first()
            )

            query = (
                session.query(Utterance.kaldi_id, Utterance.ivector)
                .filter(Utterance.ignored == False, Utterance.job_id == job.id)  # noqa
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
                yield 1
            input_proc.stdin.close()
            self.check_call(input_proc)
        engine.dispose()


@dataclassy.dataclass(slots=True)
class PldaModel:
    mean: np.ndarray
    diagonalizing_transform: np.ndarray
    psi: np.ndarray
    dimension: int
    offset: typing.Optional[np.ndarray] = None

    @classmethod
    def load(cls, plda_path):
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
                mean = np.fromstring(line, dtype="float32", sep=" ")
            elif diagonalizing_transform is None:
                if "[" in line:
                    continue
                end_mat = "]" in line
                line = line.replace("[", "").replace("]", "").strip()
                row = np.fromstring(line, dtype="float32", sep=" ")
                diagonalizing_transform_lines.append(row)
                if end_mat:
                    diagonalizing_transform = np.array(
                        diagonalizing_transform_lines, dtype="float32"
                    )
            elif psi is None:
                line = line.strip()[2:-2]
                psi = np.fromstring(line, dtype="float32", sep=" ")
        copy_proc.wait()
        return PldaModel(
            mean, diagonalizing_transform, psi, GLOBAL_CONFIG.current_profile.plda_dimension
        )

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
        ivectors = self.diagonalizing_transform @ ivectors
        l2_norm = np.linalg.norm(ivectors, axis=0, keepdims=True)
        l2_norm = l2_norm / math.sqrt(dim)

        ivectors_new = ivectors / l2_norm

        return ivectors_new.T

    def compute_pca_transform(self, ivectors: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
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
        numpy.ndarray
            Transform
        """

        num_rows = ivectors.shape[0]
        mean = np.mean(ivectors, 0, keepdims=True)
        S = np.matmul(ivectors.T, ivectors)
        S = S / num_rows

        S = S - mean.T @ mean

        ev_s, eig_s, _ = np.linalg.svd(S, full_matrices=True)
        energy_percent = np.sum(eig_s[: self.dimension]) / np.sum(eig_s)
        logger.debug(f"PLDA PCA transform energy: {energy_percent*100:.2f}%")
        transform = ev_s[:, : self.dimension]

        transxvec = ivectors @ transform
        newX = transxvec

        return newX, transform.T

    def apply_transform(self, transform_in: np.ndarray):
        """
        Adapted from https://github.com/prachiisc/PLDA_scoring/blob/master/PLDA_scoring.py#L101

        Parameters
        ----------
        transform_in : numpy.ndarray
           PCA transform
        """

        mean_plda = self.mean
        # transfomed mean vector
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
        self.mean = new_mean
        self.diagonalizing_transform = Dnew
        self.psi = psi_new
        self.offset = -Dnew @ new_mean.reshape(-1, 1)

    def transform_ivectors(self, ivectors: np.ndarray) -> np.ndarray:
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

        D = self.diagonalizing_transform
        Dnew = D.T
        X_new = ivectors @ Dnew
        X_new = X_new + offset
        # Get normalizing factor
        # Defaults : normalize_length(true), simple_length_norm(false)
        X_new_sq = X_new**2
        psi = self.psi
        inv_covar = (1.0 / (1.0 + psi)).reshape(-1, 1)
        dot_prod = X_new_sq @ inv_covar  # N X 1
        Dim = D.shape[0]
        normfactor = np.sqrt(Dim / dot_prod)
        X_new = X_new * normfactor

        return X_new
