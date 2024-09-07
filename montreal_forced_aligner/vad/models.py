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


class MfaVAD(VAD):
    def energy_VAD(
        self,
        audio_file: typing.Union[str, Path, np.ndarray, torch.Tensor],
        boundaries,
        activation_th=0.5,
        deactivation_th=0.0,
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
        boundaries: torch.Tensor
            torch.Tensor containing the speech boundaries. It can be derived using the
            get_boundaries method.
        activation_th: float
            A new speech segment is started it the energy is above activation_th.
        deactivation_th: float
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
        new_boundaries = []

        # Processing speech segments
        for i in range(boundaries.shape[0]):
            begin_sample = int(boundaries[i, 0] * sample_rate)
            end_sample = int(boundaries[i, 1] * sample_rate)
            seg_len = end_sample - begin_sample
            if seg_len < chunk_len:
                continue
            if not isinstance(audio_file, torch.Tensor):
                # Reading the speech segment
                segment, _ = torchaudio.load(
                    audio_file, frame_offset=begin_sample, num_frames=seg_len
                )
            else:
                segment = audio_file[:, begin_sample : begin_sample + seg_len]

            # Create chunks
            segment_chunks = self.create_chunks(
                segment, chunk_size=chunk_len, chunk_stride=chunk_len
            )

            # Energy computation within each chunk
            energy_chunks = segment_chunks.abs().sum(-1) + eps
            energy_chunks = energy_chunks.log()

            # Energy normalization
            energy_chunks = (
                (energy_chunks - energy_chunks.mean()) / (2 * energy_chunks.std())
            ) + 0.5
            energy_chunks = energy_chunks.unsqueeze(0).unsqueeze(2)

            # Apply threshold based on the energy value
            energy_vad = self.apply_threshold(
                energy_chunks,
                activation_th=activation_th,
                deactivation_th=deactivation_th,
            )

            # Get the boundaries
            energy_boundaries = self.get_boundaries(energy_vad, output_value="seconds")

            # Get the final boundaries in the original signal
            for j in range(energy_boundaries.shape[0]):
                start_en = boundaries[i, 0] + energy_boundaries[j, 0]
                end_end = boundaries[i, 0] + energy_boundaries[j, 1]
                new_boundaries.append([start_en, end_end])

        # Convert boundaries to tensor
        new_boundaries = torch.FloatTensor(new_boundaries).to(boundaries.device)
        return new_boundaries

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
        apply_energy_VAD: bool = False,
        double_check: bool = False,
        close_th: float = 0.333,
        len_th: float = 0.333,
        activation_th: float = 0.5,
        deactivation_th: float = 0.25,
        en_activation_th: float = 0.5,
        en_deactivation_th: float = 0.4,
        speech_th: float = 0.5,
        allow_empty: bool = True,
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
        prob_chunks = self.get_speech_prob_chunk(y).float()
        prob_th = self.apply_threshold(
            prob_chunks,
            activation_th=activation_th,
            deactivation_th=deactivation_th,
        ).float()

        # Compute the boundaries of the speech segments
        boundaries = self.get_boundaries(prob_th, output_value="seconds").cpu()
        if isinstance(segment, Segment) and segment.begin is not None:
            boundaries += segment.begin
        # Apply energy-based VAD on the detected speech segments
        if apply_energy_VAD:
            vad_boundaries = self.energy_VAD(
                y,
                boundaries,
                activation_th=en_activation_th,
                deactivation_th=en_deactivation_th,
            )
            if vad_boundaries.size(0) != 0 or allow_empty:
                boundaries = vad_boundaries

        # Merge short segments
        boundaries = self.merge_close_segments(boundaries, close_th=close_th)

        # Remove short segments
        filtered_boundaries = self.remove_short_segments(boundaries, len_th=len_th)
        if filtered_boundaries.size(0) != 0 or allow_empty:
            boundaries = filtered_boundaries

        # Double check speech segments
        if double_check:
            checked_boundaries = self.double_check_speech_segments(
                boundaries, y, speech_th=speech_th
            )
            if checked_boundaries.size(0) != 0 or allow_empty:
                boundaries = checked_boundaries
        boundaries[:, 0] -= round(close_th / 2, 3)
        boundaries[:, 1] += round(close_th / 2, 3)
        segments = []
        for i in range(boundaries.numpy().shape[0]):
            begin, end = boundaries[i]
            if i == 0:
                begin = max(begin, 0)
            if i == boundaries.numpy().shape[0] - 1:
                end = min(
                    end,
                    segment.end
                    if isinstance(segment, Segment)
                    else segment.shape[0] / self.sample_rate,
                )
            seg = Segment(
                segment.file_path if isinstance(segment, Segment) else "",
                float(begin),
                float(end),
                segment.channel if isinstance(segment, Segment) else 0,
            )
            segments.append(seg)
        return segments

    def segment_for_whisper(
        self,
        segment: typing.Union[torch.Tensor, np.ndarray],
        apply_energy_VAD: bool = False,
        close_th: float = 0.333,
        len_th: float = 0.333,
        activation_th: float = 0.5,
        deactivation_th: float = 0.25,
        en_activation_th: float = 0.5,
        en_deactivation_th: float = 0.4,
        **kwargs,
    ) -> typing.List[typing.Dict[str, float]]:
        if len(segment.shape) == 1:
            y = torch.tensor(segment[np.newaxis, :])
        elif not torch.is_tensor(segment):
            y = torch.tensor(segment)
        else:
            y = segment
        prob_chunks = self.get_speech_prob_chunk(y).float()
        prob_th = self.apply_threshold(
            prob_chunks,
            activation_th=activation_th,
            deactivation_th=deactivation_th,
        ).float()

        # Compute the boundaries of the speech segments
        boundaries = self.get_boundaries(prob_th, output_value="seconds").cpu()
        del prob_chunks
        del prob_th

        # Apply energy-based VAD on the detected speech segments
        if apply_energy_VAD:
            vad_boundaries = self.energy_VAD(
                y,
                boundaries,
                activation_th=en_activation_th,
                deactivation_th=en_deactivation_th,
            )
            boundaries = vad_boundaries

        # Merge short segments
        boundaries = self.merge_close_segments(boundaries, close_th=close_th)

        # Remove short segments
        filtered_boundaries = self.remove_short_segments(boundaries, len_th=len_th)
        if filtered_boundaries.size(0) != 0:
            boundaries = filtered_boundaries
        boundaries[:, 0] -= round(close_th / 2, 3)
        boundaries[:, 1] += round(close_th / 2, 3)
        segments = []
        for i in range(boundaries.numpy().shape[0]):
            begin, end = boundaries[i]
            if i == 0:
                begin = max(begin, 0)
            if i == boundaries.numpy().shape[0] - 1:
                end = min(end, segment.shape[0] / self.sample_rate)
            f1 = int(float(begin) * self.sample_rate)
            f2 = int(float(end) * self.sample_rate)
            segments.append({"start": float(begin), "end": float(end), "inputs": y[0, f1:f2]})
        return segments


class SpeechbrainSegmenterMixin:
    def __init__(
        self,
        apply_energy_vad: bool = False,
        double_check: bool = False,
        close_th: float = 0.333,
        len_th: float = 0.333,
        activation_th: float = 0.5,
        deactivation_th: float = 0.25,
        en_activation_th: float = 0.5,
        en_deactivation_th: float = 0.4,
        speech_th: float = 0.5,
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
        self.close_th = close_th
        self.len_th = len_th
        self.activation_th = activation_th
        self.deactivation_th = deactivation_th
        self.en_activation_th = en_activation_th
        self.en_deactivation_th = en_deactivation_th
        self.speech_th = speech_th
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
        return {
            "apply_energy_VAD": self.apply_energy_vad,
            "double_check": self.double_check,
            "activation_th": self.activation_th,
            "deactivation_th": self.deactivation_th,
            "en_activation_th": self.en_activation_th,
            "en_deactivation_th": self.en_deactivation_th,
            "speech_th": self.speech_th,
            "close_th": self.close_th,
            "len_th": self.len_th,
        }
