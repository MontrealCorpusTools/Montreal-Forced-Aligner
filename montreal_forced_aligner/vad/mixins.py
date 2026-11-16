"""Mixins for segmenters"""
from __future__ import annotations

import logging
import os
import sys
import typing

from montreal_forced_aligner import config
from montreal_forced_aligner.compat import FOUND_SPEECHBRAIN

__all__ = ["SegmenterMixin", "SpeechbrainSegmenterMixin"]

logger = logging.getLogger("mfa")

if typing.TYPE_CHECKING:
    from montreal_forced_aligner.abc import MetaDict


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
                "Could not import speechbrain, please ensure it is installed via `pip install speechbrain==1.0.3 transformers<5.0`"
            )
            sys.exit(1)
        super().__init__(**kwargs)
        from montreal_forced_aligner.vad.models import MfaVAD

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
