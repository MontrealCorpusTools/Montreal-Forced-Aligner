"""Class definitions for configuring sound file segmentation"""
from __future__ import annotations

import os

import yaml

from .base_config import BaseConfig
from .feature_config import FeatureConfig

__all__ = ["SegmentationConfig", "segmentation_yaml_to_config", "load_basic_segmentation"]


class SegmentationConfig(BaseConfig):
    """
    Class for storing segmentation configuration
    """

    def __init__(self, feature_config):
        self.use_mp = True
        self.energy_threshold = 5.5
        self.energy_mean_scale = 0.5
        self.max_segment_length = 30
        self.min_pause_duration = 0.05
        self.snap_boundary_threshold = 0.15
        self.feature_config = feature_config
        self.feature_config.use_energy = True
        self.overwrite = True

    def update(self, data: dict) -> None:
        """Update configuration parameters"""
        for k, v in data.items():
            if k == "use_mp":
                self.feature_config.use_mp = v
            if not hasattr(self, k):
                continue
            setattr(self, k, v)

    @property
    def segmentation_options(self):
        """Options for segmentation"""
        return {
            "max_segment_length": self.max_segment_length,
            "min_pause_duration": self.min_pause_duration,
            "snap_boundary_threshold": self.snap_boundary_threshold,
            "frame_shift": round(self.feature_config.frame_shift / 1000, 2),
        }


def segmentation_yaml_to_config(path: str) -> SegmentationConfig:
    """
    Helper function to load segmentation configurations

    Parameters
    ----------
    path: str
        Path to yaml file

    Returns
    -------
    :class:`~montreal_forced_aligner.config.segmentation_config.SegmentationConfig`
        Segmentation configuration
    """
    with open(path, "r", encoding="utf8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        global_params = {}
        feature_config = FeatureConfig()
        for k, v in data.items():
            if k == "features":
                feature_config.update(v)
            else:
                global_params[k] = v
        segmentation_config = SegmentationConfig(feature_config)
        segmentation_config.update(global_params)
        return segmentation_config


def load_basic_segmentation() -> SegmentationConfig:
    """
    Helper function to load the default parameters

    Returns
    -------
    :class:`~montreal_forced_aligner.config.segmentation_config.SegmentationConfig`
        Default segmentation configuration
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    segmentation_config = segmentation_yaml_to_config(
        os.path.join(base_dir, "basic_segmentation.yaml")
    )
    return segmentation_config
