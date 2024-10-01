"""Model classes for Voice Activity Detection"""
from __future__ import annotations

import logging
import os
import sys
import typing
import warnings
from pathlib import Path

import numpy as np
from kalpy.data import Segment

from montreal_forced_aligner import config
from montreal_forced_aligner.data import CtmInterval

if typing.TYPE_CHECKING:
    from montreal_forced_aligner.abc import MetaDict

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch_logger = logging.getLogger("speechbrain.utils.torch_audio_backend")
        torch_logger.setLevel(logging.ERROR)
        torch_logger = logging.getLogger("speechbrain.utils.train_logger")
        torch_logger.setLevel(logging.ERROR)
        import torch
        import torchaudio

        try:
            from speechbrain.pretrained import VAD
        except ImportError:  # speechbrain 1.0
            from speechbrain.inference.VAD import VAD

    FOUND_SPEECHBRAIN = True
except (ImportError, OSError):
    FOUND_SPEECHBRAIN = False
    VAD = object


logger = logging.getLogger("mfa")


def get_initial_segmentation(frames: np.ndarray, frame_shift: float) -> typing.List[CtmInterval]:
    """
    Compute initial segmentation over voice activity

    Parameters
    ----------
    frames: list[Union[int, str]]
        List of frames with VAD output
    frame_shift: float
        Frame shift of features in seconds

    Returns
    -------
    List[CtmInterval]
        Initial segmentation
    """
    segments = []
    cur_segment = None
    silent_frames = 0
    non_silent_frames = 0
    for i in range(frames.shape[0]):
        f = frames[i]
        if int(f) > 0:
            non_silent_frames += 1
            if cur_segment is None:
                cur_segment = CtmInterval(begin=i * frame_shift, end=0, label="speech")
        else:
            silent_frames += 1
            if cur_segment is not None:
                cur_segment.end = (i - 1) * frame_shift
                segments.append(cur_segment)
                cur_segment = None
    if cur_segment is not None:
        cur_segment.end = len(frames) * frame_shift
        segments.append(cur_segment)
    return segments


def merge_segments(
    segments: typing.List[CtmInterval],
    min_pause_duration: float,
    max_segment_length: float,
    min_segment_length: float,
    snap_boundaries: bool = True,
) -> typing.List[CtmInterval]:
    """
    Merge segments together

    Parameters
    ----------
    segments: SegmentationType
        Initial segments
    min_pause_duration: float
        Minimum amount of silence time to mark an utterance boundary
    max_segment_length: float
        Maximum length of segments before they're broken up
    min_segment_length: float
        Minimum length of segments returned

    Returns
    -------
    List[CtmInterval]
        Merged segments
    """
    merged_segments = []
    snap_boundary_threshold = 0
    if snap_boundaries:
        snap_boundary_threshold = min_pause_duration / 2
    for s in segments:
        if (
            not merged_segments
            or s.begin > merged_segments[-1].end + min_pause_duration
            or s.end - merged_segments[-1].begin > max_segment_length
        ):
            if merged_segments and snap_boundary_threshold:
                boundary_gap = s.begin - merged_segments[-1].end
                if boundary_gap < snap_boundary_threshold:
                    half_boundary = boundary_gap / 2
                else:
                    half_boundary = snap_boundary_threshold / 2
                merged_segments[-1].end += half_boundary
                s.begin -= half_boundary

            merged_segments.append(s)
        else:
            merged_segments[-1].end = s.end
    return [x for x in merged_segments if x.end - x.begin > min_segment_length]


class MfaVAD(VAD):
    def energy_VAD(
        self,
        audio_file: typing.Union[str, Path, np.ndarray, torch.Tensor],
        segments,
        activation_threshold=0.5,
        deactivation_threshold=0.0,
        eps=1e-6,
    ):
        """Applies energy-based VAD within the detected speech segments.The neural
        network VAD often creates longer segments and tends to merge segments that
        are close with each other.

        The energy VAD post-processes can be useful for having a fine-grained voice
        activity detection.

        The energy VAD computes the energy within the small chunks. The energy is
        normalized within the segment to have mean 0.5 and +-0.5 of std.
        This helps to set the energy threshold.

        Arguments
        ---------
        audio_file: path
            Path of the audio file containing the recording. The file is read
            with torchaudio.
        segments: list[CtmInterval]
            torch.Tensor containing the speech boundaries. It can be derived using the
            get_boundaries method.
        activation_threshold: float
            A new speech segment is started it the energy is above activation_th.
        deactivation_threshold: float
            The segment is considered ended when the energy is <= deactivation_th.
        eps: float
            Small constant for numerical stability.

        Returns
        -------
        new_boundaries
            The new boundaries that are post-processed by the energy VAD.
        """
        if isinstance(audio_file, (str, Path)):
            # Getting the total size of the input file
            sample_rate, audio_len = self._get_audio_info(audio_file)

            if sample_rate != self.sample_rate:
                raise ValueError(
                    "The detected sample rate is different from that set in the hparam file"
                )
        else:
            sample_rate = self.sample_rate

        # Computing the chunk length of the energy window
        chunk_len = int(self.time_resolution * sample_rate)
        new_segments = []

        # Processing speech segments
        for segment in segments:
            begin_sample = int(segment.begin * sample_rate)
            end_sample = int(segment.end * sample_rate)
            seg_len = end_sample - begin_sample
            if seg_len < chunk_len:
                continue
            if not isinstance(audio_file, torch.Tensor):
                # Reading the speech segment
                audio, _ = torchaudio.load(
                    audio_file, frame_offset=begin_sample, num_frames=seg_len
                )
            else:
                audio = audio_file[:, begin_sample : begin_sample + seg_len]

            # Create chunks
            segment_chunks = self.create_chunks(
                audio, chunk_size=chunk_len, chunk_stride=chunk_len
            )

            # Energy computation within each chunk
            energy_chunks = segment_chunks.abs().sum(-1) + eps
            energy_chunks = energy_chunks.log()

            # Energy normalization
            energy_chunks = (
                (energy_chunks - energy_chunks.mean()) / (2 * energy_chunks.std())
            ) + 0.5
            energy_chunks = energy_chunks

            # Apply threshold based on the energy value
            new_segments.extend(
                self.generate_segments(
                    energy_chunks,
                    activation_threshold=activation_threshold,
                    deactivation_threshold=deactivation_threshold,
                    begin=segment.begin,
                    end=segment.end,
                )
            )
        return new_segments

    def double_check_speech_segments(self, boundaries, audio_file, speech_th=0.5):
        """Takes in input the boundaries of the detected speech segments and
        double checks (using the neural VAD) that they actually contain speech.

        Arguments
        ---------
        boundaries: torch.Tensor
            torch.Tensor containing the boundaries of the speech segments.
        audio_file: path
            The original audio file used to compute vad_out.
        speech_th: float
            Threshold on the mean posterior probability over which speech is
            confirmed. Below that threshold, the segment is re-assigned to a
            non-speech region.

        Returns
        -------
        new_boundaries
            The boundaries of the segments where speech activity is confirmed.
        """

        if isinstance(audio_file, (str, Path)):
            # Getting the total size of the input file
            sample_rate, audio_len = self._get_audio_info(audio_file)

            if sample_rate != self.sample_rate:
                raise ValueError(
                    "The detected sample rate is different from that set in the hparam file"
                )
        else:
            sample_rate = self.sample_rate

        # Double check the segments
        new_boundaries = []
        for i in range(boundaries.shape[0]):
            beg_sample = int(boundaries[i, 0] * sample_rate)
            end_sample = int(boundaries[i, 1] * sample_rate)
            len_seg = end_sample - beg_sample

            if not isinstance(audio_file, torch.Tensor):
                # Read the candidate speech segment
                segment, fs = torchaudio.load(
                    str(audio_file), frame_offset=beg_sample, num_frames=len_seg
                )
            else:
                segment = audio_file[:, beg_sample : beg_sample + len_seg]
            speech_prob = self.get_speech_prob_chunk(segment)
            if speech_prob.mean() > speech_th:
                # Accept this as a speech segment
                new_boundaries.append([boundaries[i, 0], boundaries[i, 1]])

        # Convert boundaries from list to tensor
        new_boundaries = torch.FloatTensor(new_boundaries).to(boundaries.device)
        return new_boundaries

    def segment_utterance(
        self,
        segment: typing.Union[Segment, np.ndarray],
        apply_energy_vad: bool = False,
        min_pause_duration: float = 0.333,
        max_segment_length: float = 30,
        min_segment_length: float = 0.333,
        activation_threshold: float = 0.5,
        deactivation_threshold: float = 0.25,
        energy_activation_threshold: float = 0.5,
        energy_deactivation_threshold: float = 0.4,
        **kwargs,
    ) -> typing.List[Segment]:
        if isinstance(segment, Segment):
            y = torch.tensor(segment.wave[np.newaxis, :])
        else:
            if len(segment.shape) == 1:
                y = torch.tensor(segment[np.newaxis, :])
            elif not torch.is_tensor(segment):
                y = torch.tensor(segment)
            else:
                y = segment
        prob_chunks = self.get_speech_prob_chunk(y).float().cpu().numpy()[0, ...]
        # Compute the boundaries of the speech segments
        segments = self.generate_segments(
            prob_chunks,
            activation_threshold=activation_threshold,
            deactivation_threshold=deactivation_threshold,
            begin=segment.begin
            if isinstance(segment, Segment) and segment.begin is not None
            else None,
            end=segment.end if isinstance(segment, Segment) and segment.end is not None else None,
        )

        # Apply energy-based VAD on the detected speech segments
        if apply_energy_vad:
            segments = self.energy_VAD(
                y,
                segments,
                activation_threshold=energy_activation_threshold,
                deactivation_threshold=energy_deactivation_threshold,
            )

        # Merge short segments
        segments = merge_segments(
            segments,
            min_pause_duration=min_pause_duration,
            max_segment_length=max_segment_length,
            min_segment_length=min_segment_length,
            snap_boundaries=False,
        )

        # Padding
        for i, s in enumerate(segments):
            begin, end = s.begin, s.end
            begin -= min_pause_duration / 2
            end += min_pause_duration / 2
            if i == 0:
                begin = max(begin, 0)
            if i == len(segments) - 1:
                end = min(
                    end,
                    segment.shape[0] / self.sample_rate
                    if not isinstance(segment, Segment)
                    else segment.end,
                )
            s.begin = begin
            s.end = end
            if isinstance(segment, Segment):
                segments[i] = Segment(segment.file_path, s.begin, s.end, segment.channel)
        return segments

    def generate_segments(
        self, vad_prob, activation_threshold=0.5, deactivation_threshold=0.25, begin=None, end=None
    ):
        """Scans the frame-level speech probabilities and applies a threshold
        on them. Speech starts when a value larger than activation_th is
        detected, while it ends when observing a value lower than
        the deactivation_th.

        Arguments
        ---------
        vad_prob: numpy.ndarray
            Frame-level speech probabilities.
        activation_threshold:  float
            Threshold for starting a speech segment.
        deactivation_threshold: float
            Threshold for ending a speech segment.

        Returns
        -------
        vad_th: torch.Tensor
            torch.Tensor containing 1 for speech regions and 0 for non-speech regions.
        """
        if begin is None:
            begin = 0
        # Loop over batches and time steps
        is_active = vad_prob[0] > activation_threshold
        start = 0
        boundaries = []
        for time_step in range(1, vad_prob.shape[0] - 1):
            y = vad_prob[time_step]
            if is_active:
                if y < deactivation_threshold:
                    e = self.time_resolution * (time_step - 1)
                    boundaries.append(
                        CtmInterval(begin=start + begin, end=e + begin, label="speech")
                    )
                    is_active = False
            elif y > activation_threshold:
                is_active = True
                start = self.time_resolution * time_step
        if is_active:
            if end is not None:
                e = end
            else:
                e = self.time_resolution * vad_prob.shape[0]
                e += begin
            boundaries.append(CtmInterval(begin=start + begin, end=e, label="speech"))
        return boundaries

    def get_speech_prob_chunk(self, wavs, wav_lens=None):
        """Outputs the frame-level posterior probability for the input audio chunks
        Outputs close to zero refers to time steps with a low probability of speech
        activity, while outputs closer to one likely contain speech.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        torch.Tensor
            The encoded batch
        """
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        # Storing waveform in the specified device
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()

        # Computing features and embeddings
        feats = self.mods.compute_features(wavs)
        feats = self.mods.mean_var_norm(feats, wav_lens)
        outputs = self.mods.cnn(feats)

        outputs = outputs.reshape(
            outputs.shape[0],
            outputs.shape[1],
            outputs.shape[2] * outputs.shape[3],
        )

        outputs, h = self.mods.rnn(outputs)
        outputs = self.mods.dnn(outputs)
        output_prob = torch.sigmoid(outputs)

        return output_prob

    def segment_for_whisper(
        self,
        segment: typing.Union[torch.Tensor, np.ndarray],
        apply_energy_vad: bool = True,
        max_segment_length: float = 30,
        min_segment_length: float = 0.333,
        min_pause_duration: float = 0.333,
        activation_threshold: float = 0.5,
        deactivation_threshold: float = 0.25,
        en_activation_threshold: float = 0.5,
        en_deactivation_threshold: float = 0.4,
        **kwargs,
    ) -> typing.List[typing.Dict[str, float]]:
        if isinstance(segment, Segment):
            y = torch.tensor(segment.wave[np.newaxis, :])
        else:
            if len(segment.shape) == 1:
                y = torch.tensor(segment[np.newaxis, :])
            elif not torch.is_tensor(segment):
                y = torch.tensor(segment)
            else:
                y = segment
        segments = self.segment_utterance(
            segment,
            apply_energy_vad=apply_energy_vad,
            max_segment_length=max_segment_length,
            min_segment_length=min_segment_length,
            min_pause_duration=min_pause_duration,
            activation_threshold=activation_threshold,
            deactivation_threshold=deactivation_threshold,
            en_activation_threshold=en_activation_threshold,
            en_deactivation_threshold=en_deactivation_threshold,
            **kwargs,
        )

        # Padding
        segments_for_whisper = []
        for i, s in enumerate(segments):
            begin, end = s.begin, s.end
            f1 = int(round(begin, 3) * self.sample_rate)
            f2 = int(round(end, 3) * self.sample_rate)
            segments_for_whisper.append(
                {"start": float(begin), "end": float(end), "inputs": y[0, f1:f2]}
            )
        return segments_for_whisper


class SegmenterMixin:
    def __init__(
        self,
        max_segment_length: float = 30,
        min_segment_length: float = 0.333,
        min_pause_duration: float = 0.333,
        activation_threshold: float = 0.5,
        deactivation_threshold: float = 0.25,
        energy_activation_threshold: float = 0.5,
        energy_deactivation_threshold: float = 0.4,
        **kwargs,
    ):
        self.max_segment_length = max_segment_length
        self.min_segment_length = min_segment_length
        self.min_pause_duration = min_pause_duration
        self.activation_threshold = activation_threshold
        self.deactivation_threshold = deactivation_threshold
        self.energy_activation_threshold = energy_activation_threshold
        self.energy_deactivation_threshold = energy_deactivation_threshold
        super().__init__(**kwargs)

    @property
    def segmentation_options(self) -> MetaDict:
        """Options for segmentation"""
        return {
            "max_segment_length": self.max_segment_length,
            "min_segment_length": self.min_segment_length,
            "activation_threshold": self.activation_threshold,
            "deactivation_threshold": self.deactivation_threshold,
            "energy_activation_threshold": self.energy_activation_threshold,
            "energy_deactivation_threshold": self.energy_deactivation_threshold,
            "min_pause_duration": self.min_pause_duration,
        }


class SpeechbrainSegmenterMixin(SegmenterMixin):
    def __init__(
        self,
        apply_energy_vad: bool = True,
        double_check: bool = False,
        speech_threshold: float = 0.5,
        cuda: bool = False,
        **kwargs,
    ):
        if not FOUND_SPEECHBRAIN:
            logger.error(
                "Could not import speechbrain, please ensure it is installed via `pip install speechbrain`"
            )
            sys.exit(1)
        super().__init__(**kwargs)
        self.apply_energy_vad = apply_energy_vad
        self.double_check = double_check
        self.speech_threshold = speech_threshold
        self.cuda = cuda
        self.speechbrain = True
        self.vad_model = None
        model_dir = os.path.join(config.TEMPORARY_DIRECTORY, "models", "VAD")
        os.makedirs(model_dir, exist_ok=True)
        run_opts = None
        if self.cuda:
            run_opts = {"device": "cuda"}
        self.vad_model = MfaVAD.from_hparams(
            source="speechbrain/vad-crdnn-libriparty", savedir=model_dir, run_opts=run_opts
        )

    @property
    def segmentation_options(self) -> MetaDict:
        """Options for segmentation"""
        options = super().segmentation_options
        options.update(
            {
                "apply_energy_vad": self.apply_energy_vad,
                "double_check": self.double_check,
                "speech_threshold": self.speech_threshold,
            }
        )
        return options
