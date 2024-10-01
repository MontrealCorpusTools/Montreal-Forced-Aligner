"""Classes for configuring feature generation"""
from __future__ import annotations

import logging
import os
import typing
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import dataclassy
from _kalpy.feat import paste_feats
from _kalpy.matrix import CompressedMatrix, FloatVector
from _kalpy.util import BaseFloatMatrixWriter, BaseFloatVectorWriter, CompressedMatrixWriter
from kalpy.data import KaldiMapping, MatrixArchive, Segment
from kalpy.feat.data import FeatureArchive
from kalpy.feat.fmllr import FmllrComputer
from kalpy.feat.mfcc import MfccComputer
from kalpy.feat.pitch import PitchComputer
from kalpy.feat.vad import VadComputer
from kalpy.gmm.data import AlignmentArchive
from kalpy.ivector.extractor import IvectorExtractor
from kalpy.utils import generate_write_specifier
from sqlalchemy.orm import joinedload

from montreal_forced_aligner import config
from montreal_forced_aligner.abc import KaldiFunction
from montreal_forced_aligner.data import MfaArguments
from montreal_forced_aligner.db import File, Job, Phone, SoundFile, Utterance
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.utils import thread_logger

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
    "ExportIvectorsFunction",
    "ExportIvectorsArguments",
]

logger = logging.getLogger("mfa")


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class VadArguments(MfaArguments):
    """Arguments for :class:`~montreal_forced_aligner.corpus.features.ComputeVadFunction`"""

    vad_options: MetaDict


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class MfccArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.corpus.features.MfccFunction`
    """

    data_directory: Path
    mfcc_computer: MfccComputer
    pitch_computer: typing.Optional[PitchComputer]


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class FinalFeatureArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.corpus.features.FinalFeatureFunction`
    """

    data_directory: Path
    uses_cmvn: bool
    sliding_cmvn: bool
    voiced_only: bool
    subsample_feats: int


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class PitchArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.corpus.features.MfccFunction`
    """

    data_directory: Path
    pitch_options: MetaDict


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class PitchRangeArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.corpus.features.MfccFunction`
    """

    data_directory: Path
    pitch_options: MetaDict


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class CalcFmllrArguments(MfaArguments):
    """Arguments for :class:`~montreal_forced_aligner.corpus.features.CalcFmllrFunction`"""

    working_directory: Path
    ali_model_path: Path
    model_path: Path
    fmllr_options: MetaDict


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class ExtractIvectorsArguments(MfaArguments):
    """Arguments for :class:`~montreal_forced_aligner.corpus.features.ExtractIvectorsFunction`"""

    ivector_options: MetaDict
    ivector_extractor_path: Path
    ivectors_scp_path: Path
    dubm_path: Path


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

    def __init__(self, args: MfccArguments):
        super().__init__(args)
        self.data_directory = args.data_directory
        self.pitch_computer = args.pitch_computer
        self.mfcc_computer = args.mfcc_computer

    def _run(self):
        """Run the function"""
        with self.session() as session, thread_logger(
            "kalpy.mfcc", self.log_path, job_name=self.job_name
        ) as mfcc_logger:
            mfcc_logger.debug(f"MFCC parameters: {self.mfcc_computer.parameters}")
            job: typing.Optional[Job] = session.get(Job, self.job_name)
            raw_ark_path = job.construct_path(self.data_directory, "feats", "ark")
            raw_pitch_ark_path = job.construct_path(self.data_directory, "pitch", "ark")
            if raw_ark_path.exists():
                return
            limit = 10000
            offset = 0
            min_length = 0.1
            mfcc_specifier = generate_write_specifier(raw_ark_path, True)
            pitch_specifier = generate_write_specifier(raw_pitch_ark_path, True)
            mfcc_writer = CompressedMatrixWriter(mfcc_specifier)
            pitch_writer = None
            if self.pitch_computer is not None:
                mfcc_logger.debug(f"Pitch parameters: {self.pitch_computer.parameters}")
                pitch_writer = CompressedMatrixWriter(pitch_specifier)
            num_done = 0
            num_error = 0
            while True:
                utterances = (
                    session.query(Utterance, SoundFile)
                    .join(Utterance.file)
                    .join(File.sound_file)
                    .filter(
                        Utterance.job_id == self.job_name,
                        Utterance.duration >= min_length,
                    )
                    .order_by(Utterance.kaldi_id)
                    .limit(limit)
                    .offset(offset)
                )
                if utterances.count() == 0:
                    break
                for u, sf in utterances:
                    seg = Segment(str(sf.sound_file_path), u.begin, u.end, u.channel)
                    mfcc_logger.info(f"Processing {u.kaldi_id}")
                    try:
                        mfccs = self.mfcc_computer.compute_mfccs_for_export(seg, compress=True)
                    except Exception as e:
                        mfcc_logger.warning(str(e))
                        num_error += 1
                        continue

                    mfcc_writer.Write(u.kaldi_id, mfccs)
                    if self.pitch_computer is not None:
                        pitch = self.pitch_computer.compute_pitch_for_export(seg, compress=True)
                        pitch_writer.Write(u.kaldi_id, pitch)
                    num_done += 1
                    self.callback(1)
                offset += limit
            mfcc_writer.Close()
            if self.pitch_computer is not None:
                pitch_writer.Close()
            mfcc_logger.info(f"Done {num_done} utterances, errors on {num_error}.")


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

    def __init__(self, args: FinalFeatureArguments):
        super().__init__(args)
        self.data_directory = args.data_directory
        self.voiced_only = args.voiced_only
        self.uses_cmvn = args.uses_cmvn
        self.sliding_cmvn = args.sliding_cmvn
        self.subsample_feats = args.subsample_feats

    def _run(self) -> None:
        """Run the function"""
        with self.session() as session, thread_logger(
            "kalpy.mfcc", self.log_path, job_name=self.job_name
        ) as mfcc_logger:
            job: typing.Optional[Job] = session.get(Job, self.job_name)
            utterances = (
                session.query(Utterance.kaldi_id, Utterance.speaker_id)
                .filter(Utterance.job_id == self.job_name)
                .order_by(Utterance.kaldi_id)
            )
            spk2utt = KaldiMapping(list_mapping=True)
            utt2spk = KaldiMapping()
            for utt_id, speaker_id in utterances:
                utt_id = str(utt_id)
                speaker_id = str(speaker_id)
                utt2spk[utt_id] = speaker_id
                if speaker_id not in spk2utt:
                    spk2utt[speaker_id] = []
                spk2utt[speaker_id].append(utt_id)
            feats_scp_path = job.construct_path(self.data_directory, "feats", "scp")
            cmvn_scp_path = job.construct_path(self.data_directory, "cmvn", "scp")
            pitch_scp_path = job.construct_path(self.data_directory, "pitch", "scp")
            pitch_ark_path = job.construct_path(self.data_directory, "pitch", "ark")
            vad_scp_path = job.construct_path(self.data_directory, "vad", "scp")
            if not self.voiced_only or not os.path.exists(vad_scp_path):
                vad_scp_path = None
            raw_ark_path = job.construct_path(self.data_directory, "feats", "ark")
            temp_ark_path = job.construct_path(self.data_directory, "final_features", "ark")
            temp_scp_path = job.construct_path(self.data_directory, "final_features", "scp")
            write_specifier = generate_write_specifier(temp_ark_path, write_scp=True)
            feature_writer = CompressedMatrixWriter(write_specifier)
            num_done = 0
            num_error = 0
            if self.uses_cmvn:
                if not self.sliding_cmvn:
                    mfcc_archive = FeatureArchive(
                        feats_scp_path,
                        utt2spk=utt2spk,
                        cmvn_file_name=cmvn_scp_path,
                        vad_file_name=vad_scp_path,
                        subsample_n=self.subsample_feats,
                    )
                else:
                    mfcc_archive = FeatureArchive(
                        feats_scp_path,
                        utt2spk=utt2spk,
                        vad_file_name=vad_scp_path,
                        subsample_n=self.subsample_feats,
                        use_sliding_cmvn=True,
                    )
            else:
                mfcc_archive = FeatureArchive(feats_scp_path)
            if os.path.exists(pitch_scp_path):
                pitch_archive = FeatureArchive(
                    pitch_scp_path, vad_file_name=vad_scp_path, subsample_n=self.subsample_feats
                )
                for (utt_id, mfccs), (utt_id2, pitch) in zip(mfcc_archive, pitch_archive):
                    assert utt_id == utt_id2
                    try:
                        feats = paste_feats([mfccs, pitch], 1)
                    except Exception as e:
                        mfcc_logger.warning(f"Exception encountered: {e}")
                        num_error += 1
                        continue
                    mfcc_logger.info(
                        f"Processing {utt_id}: MFCC len = {mfccs.NumRows()}, "
                        f"Pitch len = {pitch.NumRows()}, Combined len = {feats.NumRows()}"
                    )
                    feats = CompressedMatrix(feats)
                    feature_writer.Write(utt_id, feats)
                    num_done += 1
                    self.callback(1)
                pitch_archive.close()
            else:
                for utt_id, mfccs in mfcc_archive:
                    mfcc_logger.info(f"Processing {utt_id}: len = {mfccs.NumRows()}")
                    mfccs = CompressedMatrix(mfccs)
                    feature_writer.Write(utt_id, mfccs)
                    num_done += 1
                    self.callback(1)
            feature_writer.Close()
            mfcc_archive.close()
            raw_ark_path.unlink()
            if pitch_scp_path.exists():
                pitch_ark_path.unlink()
                pitch_scp_path.unlink()
            feats_scp_path.unlink()
            temp_scp_path.rename(feats_scp_path)
            mfcc_logger.info(f"Done {num_done} utterances, errors on {num_error}.")


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

    def __init__(self, args: VadArguments):
        super().__init__(args)
        self.vad_options = args.vad_options

    def _run(self) -> None:
        """Run the function"""

        with self.session() as session, thread_logger(
            "kalpy.vad", self.log_path, job_name=self.job_name
        ):
            job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True))
                .filter(Job.id == self.job_name)
                .first()
            )
            vad_ark_path = job.construct_path(job.corpus.split_directory, "vad", "ark")
            feature_archive = job.construct_feature_archive(job.corpus.split_directory)
            computer = VadComputer(**self.vad_options)
            computer.export_vad(
                vad_ark_path, feature_archive, write_scp=True, callback=self.callback
            )


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

    def __init__(self, args: CalcFmllrArguments):
        super().__init__(args)
        self.working_directory = args.working_directory
        self.ali_model_path = args.ali_model_path
        self.model_path = args.model_path
        self.fmllr_options = args.fmllr_options

    def _run(self) -> None:
        """Run the function"""
        from montreal_forced_aligner.db import Dictionary

        with self.session() as session, thread_logger(
            "kalpy.fmllr", self.log_path, job_name=self.job_name
        ) as fmllr_logger:
            job: typing.Optional[Job] = session.get(
                Job, self.job_name, options=[joinedload(Job.dictionaries), joinedload(Job.corpus)]
            )

            for dict_id in job.dictionary_ids:
                d = session.get(Dictionary, dict_id)
                silence_phones = [
                    x
                    for x, in session.query(Phone.mapping_id).filter(
                        Phone.phone.in_([d.optional_silence_phone, d.oov_phone])
                    )
                ]
                fmllr_trans_path = job.construct_path(
                    job.corpus.current_subset_directory, "trans", "scp", dictionary_id=dict_id
                )
                previous_transform_archive = None
                if not fmllr_trans_path.exists():
                    fmllr_logger.debug("Computing transforms from scratch")
                    fmllr_trans_path = None
                else:
                    fmllr_logger.debug(f"Updating previous transforms {fmllr_trans_path}")
                    previous_transform_archive = MatrixArchive(fmllr_trans_path)
                spk2utt_path = job.construct_path(
                    job.corpus.current_subset_directory, "spk2utt", "scp", dictionary_id=dict_id
                )
                spk2utt = KaldiMapping(list_mapping=True)
                spk2utt.load(spk2utt_path)
                feature_archive = job.construct_feature_archive(self.working_directory, dict_id)

                fmllr_logger.debug("Feature Archive information:")
                fmllr_logger.debug(f"CMVN: {feature_archive.cmvn_read_specifier}")
                fmllr_logger.debug(f"Deltas: {feature_archive.use_deltas}")
                fmllr_logger.debug(f"Splices: {feature_archive.use_splices}")
                fmllr_logger.debug(f"LDA: {feature_archive.lda_mat_file_name}")
                fmllr_logger.debug(f"fMLLR: {feature_archive.transform_read_specifier}")
                fmllr_logger.debug("Model information:")
                fmllr_logger.debug(f"Align model path: {self.ali_model_path}")
                fmllr_logger.debug(f"Model path: {self.model_path}")

                computer = FmllrComputer(
                    self.ali_model_path,
                    self.model_path,
                    silence_phones,
                    spk2utt=spk2utt,
                    **self.fmllr_options,
                )
                ali_path = job.construct_path(self.working_directory, "ali", "ark", dict_id)
                if not ali_path.exists():
                    continue
                fmllr_logger.debug(f"Alignment path: {ali_path}")
                alignment_archive = AlignmentArchive(ali_path)
                temp_trans_path = job.construct_path(
                    self.working_directory, "trans", "ark", dict_id
                )
                computer.export_transforms(
                    temp_trans_path,
                    feature_archive,
                    alignment_archive,
                    previous_transform_archive=previous_transform_archive,
                    callback=self.callback,
                )
                feature_archive.close()
                del previous_transform_archive
                del feature_archive
                del alignment_archive
                del computer
                if fmllr_trans_path is not None:
                    fmllr_trans_path.unlink()
                    fmllr_trans_path.with_suffix(".ark").unlink()
                trans_archive = MatrixArchive(temp_trans_path)
                write_specifier = generate_write_specifier(
                    job.construct_path(
                        job.corpus.current_subset_directory, "trans", "ark", dictionary_id=dict_id
                    ),
                    write_scp=True,
                )
                writer = BaseFloatMatrixWriter(write_specifier)
                for speaker, trans in trans_archive:
                    writer.Write(str(speaker), trans)
                writer.Close()
                del trans_archive
                temp_trans_path.unlink()


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
        raw_energy: bool = False,
        frame_shift: int = 10,
        frame_length: int = 25,
        snip_edges: bool = False,
        low_frequency: int = 20,
        high_frequency: int = 7800,
        sample_frequency: int = 16000,
        allow_downsample: bool = True,
        allow_upsample: bool = True,
        dither: float = 0.0,
        energy_floor: float = 0.0,
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
        self.raw_energy = raw_energy
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
        self.mfcc_computer = MfccComputer(**self.mfcc_options)
        self.pitch_computer = None
        if self.use_pitch:
            self.pitch_computer = PitchComputer(**self.pitch_options)

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
    def working_directory(self) -> Path:
        """Abstract method for working directory"""
        raise NotImplementedError

    @property
    def corpus_output_directory(self) -> str:
        """Abstract method for working directory of corpus"""
        raise NotImplementedError

    @property
    def data_directory(self) -> str:
        """Abstract method for corpus data directory"""
        raise NotImplementedError

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

    def calc_fmllr(self) -> None:
        """Abstract method for calculating fMLLR transforms"""
        raise NotImplementedError

    @property
    def fmllr_options(self) -> MetaDict:
        """Options for use in calculating fMLLR transforms"""
        return {
            "fmllr_update_type": self.fmllr_update_type,
            "silence_weight": self.silence_weight,
            "acoustic_scale": getattr(self, "acoustic_scale", 0.1),
        }

    @property
    def lda_options(self) -> MetaDict:
        """Options for computing LDA"""
        if getattr(self, "acoustic_model", None) is not None:
            return self.acoustic_model.lda_options
        if getattr(self, "ivector_extractor", None) is not None:
            return self.ivector_extractor.lda_options
        return {
            "splice_left_context": self.splice_left_context,
            "splice_right_context": self.splice_right_context,
        }

    @property
    def mfcc_options(self) -> MetaDict:
        """Parameters to use in computing MFCC features."""
        if getattr(self, "acoustic_model", None) is not None:
            options = self.acoustic_model.mfcc_options
        elif getattr(self, "ivector_extractor", None) is not None:
            options = self.ivector_extractor.mfcc_options
        else:
            options = {
                "use_energy": self.use_energy,
                "raw_energy": self.raw_energy,
                "dither": self.dither,
                "energy_floor": self.energy_floor,
                "num_coefficients": self.num_coefficients,
                "num_mel_bins": self.num_mel_bins,
                "cepstral_lifter": self.cepstral_lifter,
                "preemphasis_coefficient": self.preemphasis_coefficient,
                "frame_shift": self.frame_shift,
                "frame_length": self.frame_length,
                "low_frequency": self.low_frequency,
                "high_frequency": self.high_frequency,
                "sample_frequency": self.sample_frequency,
                "allow_downsample": self.allow_downsample,
                "allow_upsample": self.allow_upsample,
                "snip_edges": self.snip_edges,
            }
        options.update(
            {
                "dither": 0.0001,
                "energy_floor": 1.0,
            }
        )
        options.update(
            {
                "dither": self.dither,
                "energy_floor": self.energy_floor,
                "snip_edges": self.snip_edges,
                "frame_shift": self.frame_shift,
            }
        )
        return options

    @property
    def pitch_options(self) -> MetaDict:
        """Parameters to use in computing pitch features."""
        if getattr(self, "acoustic_model", None) is not None:
            options = self.acoustic_model.pitch_options
        elif getattr(self, "ivector_extractor", None) is not None:
            options = self.ivector_extractor.pitch_options
        else:
            use_pitch = self.use_pitch
            use_voicing = self.use_voicing
            use_delta_pitch = self.use_delta_pitch
            normalize = self.normalize_pitch
            options = {
                "frame_shift": self.frame_shift,
                "frame_length": self.frame_length,
                "min_f0": self.min_f0,
                "max_f0": self.max_f0,
                "sample_frequency": self.sample_frequency,
                "penalty_factor": self.penalty_factor,
                "delta_pitch": self.delta_pitch,
                "snip_edges": self.snip_edges,
                "add_normalized_log_pitch": False,
                "add_delta_pitch": False,
                "add_pov_feature": False,
            }
            if use_pitch:
                options["add_normalized_log_pitch"] = normalize
                options["add_raw_log_pitch"] = not normalize
            options["add_delta_pitch"] = use_delta_pitch
            options["add_pov_feature"] = use_voicing
        options.update(
            {
                "min_f0": self.min_f0,
                "max_f0": self.max_f0,
                "snip_edges": self.snip_edges,
                "frame_shift": self.frame_shift,
            }
        )
        return options


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
        self.ivector_dimension = config.IVECTOR_DIMENSION
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

    def __init__(self, args: ExtractIvectorsArguments):
        super().__init__(args)
        self.ivector_options = args.ivector_options
        self.ivector_extractor_path = args.ivector_extractor_path
        self.ivectors_scp_path = args.ivectors_scp_path
        self.dubm_path = args.dubm_path

    def _run(self):
        """Run the function"""
        if os.path.exists(self.ivectors_scp_path):
            return
        with self.session() as session, thread_logger(
            "kalpy.ivector", self.log_path, job_name=self.job_name
        ):
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True))
                .filter(Job.id == self.job_name)
                .first()
            )
            ivector_ark_path = self.ivectors_scp_path.with_suffix(".ark")
            feature_archive = job.construct_feature_archive(job.corpus.current_subset_directory)
            ivector_extractor = IvectorExtractor(
                self.dubm_path,
                self.ivector_extractor_path,
                acoustic_weight=self.ivector_options["posterior_scale"],
                max_count=self.ivector_options["max_count"],
                num_gselect=self.ivector_options["num_gselect"],
                min_post=self.ivector_options["min_post"],
            )
            ivector_extractor.export_ivectors(
                ivector_ark_path, feature_archive, write_scp=True, callback=self.callback
            )


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

    def _run(self):
        """Run the function"""
        with self.session() as session:
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

            writer = BaseFloatVectorWriter(
                generate_write_specifier(ivector_ark_path, write_scp=True)
            )
            for utt_id, ivector in query:
                if ivector is None:
                    continue
                kaldi_ivector = FloatVector()
                kaldi_ivector.from_numpy(ivector)
                writer.Write(utt_id, kaldi_ivector)
            writer.Close()
            with mfa_open(ivector_scp_path) as f:
                for line in f:
                    line = line.strip()
                    utt_id, ark_path = line.split(maxsplit=1)
                    utt_id = int(utt_id.split("-")[1])
                    self.callback((utt_id, ark_path))
