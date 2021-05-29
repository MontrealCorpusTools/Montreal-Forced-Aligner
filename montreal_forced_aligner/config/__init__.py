import os
from .base_config import BaseConfig, save_config, ConfigError
from .align_config import AlignConfig, load_basic_align, align_yaml_to_config, FeatureConfig
from .speaker_classification_config import SpeakerClassificationConfig, classification_yaml_to_config, load_basic_classification
from .train_config import TrainingConfig, load_basic_train, load_basic_train_ivector, load_test_config, \
    train_yaml_to_config
from .train_lm_config import TrainLMConfig, load_basic_train_lm, train_lm_yaml_to_config
from .transcribe_config import TranscribeConfig, load_basic_transcribe, transcribe_yaml_to_config
from .segmentation_config import SegmentationConfig, segmentation_yaml_to_config, load_basic_segmentation

TEMP_DIR = os.path.expanduser('~/Documents/MFA')




