"""
Functions for compatibility with other libraries and optional dependencies
"""
import importlib.util
import logging
import warnings

FOUND_HDBSCAN = importlib.util.find_spec("hdbscan") is not None
FOUND_SPEECHBRAIN = importlib.util.find_spec("speechbrain") is not None
FOUND_PYANNOTE = importlib.util.find_spec("pyannote") is not None
FOUND_WHISPERX = importlib.util.find_spec("whisperx") is not None
FOUND_TORCH = importlib.util.find_spec("torch") is not None

SPACY_AVAILABLE = importlib.util.find_spec("spacy") is not None
TH_AVAILABLE = importlib.util.find_spec("pythainlp") is not None
KO_AVAILABLE = importlib.util.find_spec("mecab") is not None
JA_AVAILABLE = importlib.util.find_spec("sudachipy") is not None
ZH_AVAILABLE = (
    importlib.util.find_spec("hanziconv") is not None
    and importlib.util.find_spec("spacy_pkuseg") is not None
    and importlib.util.find_spec("dragonmapper") is not None
)


def import_speechbrain_transcription_modules():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch_logger = logging.getLogger("speechbrain.utils.torch_audio_backend")
        torch_logger.setLevel(logging.ERROR)
        torch_logger = logging.getLogger("speechbrain.utils.train_logger")
        torch_logger.setLevel(logging.ERROR)
        transformers_logger = logging.getLogger("transformers.modeling_utils")
        transformers_logger.setLevel(logging.ERROR)
        transformers_logger = logging.getLogger(
            "speechbrain.lobes.models.huggingface_transformers.huggingface"
        )
        transformers_logger.setLevel(logging.ERROR)
        try:
            from speechbrain.pretrained import EncoderASR, WhisperASR
        except (ImportError, ModuleNotFoundError):  # speechbrain 1.0
            from speechbrain.inference.ASR import EncoderASR, WhisperASR
        return EncoderASR, WhisperASR


def import_speechbrain_diarization_modules():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        speechbrain_logger = logging.getLogger("speechbrain.utils.torch_audio_backend")
        speechbrain_logger.setLevel(logging.ERROR)
        speechbrain_logger = logging.getLogger("speechbrain.utils.train_logger")
        speechbrain_logger.setLevel(logging.ERROR)
        try:
            from speechbrain.pretrained import EncoderClassifier, SpeakerRecognition
        except (ImportError, ModuleNotFoundError):  # speechbrain 1.0
            from speechbrain.inference.classifiers import EncoderClassifier
            from speechbrain.inference.speaker import SpeakerRecognition
    return EncoderClassifier, SpeakerRecognition
