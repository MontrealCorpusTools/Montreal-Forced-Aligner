"""Utility functions for command line commands"""
from __future__ import annotations

import os

import yaml

from ..exceptions import (
    FileArgumentNotFoundError,
    ModelExtensionError,
    ModelTypeNotSupportedError,
    NoDefaultSpeakerDictionaryError,
    PretrainedModelNotFoundError,
)
from ..models import MODEL_TYPES

__all__ = ["validate_model_arg"]


def validate_model_arg(name: str, model_type: str) -> str:
    """
    Validate pretrained model name argument

    Parameters
    ----------
    name: str
        Name of model
    model_type: str
        Type of model

    Returns
    -------
    str
        Full path of validated model

    Raises
    ------
    :class:`~montreal_forced_aligner.exceptions.ModelTypeNotSupportedError`
        If the type of model is not supported
    :class:`~montreal_forced_aligner.exceptions.FileArgumentNotFoundError`
        If the file specified is not found
    :class:`~montreal_forced_aligner.exceptions.PretrainedModelNotFoundError`
        If the pretrained model specified is not found
    :class:`~montreal_forced_aligner.exceptions.ModelExtensionError`
        If the extension is not valid for the specified model type
    :class:`~montreal_forced_aligner.exceptions.NoDefaultSpeakerDictionaryError`
        If a multispeaker dictionary does not have a default dictionary
    """
    if model_type not in MODEL_TYPES:
        raise ModelTypeNotSupportedError(model_type, MODEL_TYPES)
    model_class = MODEL_TYPES[model_type]
    available_models = model_class.get_available_models()
    model_class = MODEL_TYPES[model_type]
    if name in available_models:
        name = model_class.get_pretrained_path(name)
    elif model_class.valid_extension(name):
        if not os.path.exists(name):
            raise FileArgumentNotFoundError(name)
        if model_type == "dictionary" and os.path.splitext(name)[1].lower() == ".yaml":
            with open(name, "r", encoding="utf8") as f:
                data = yaml.safe_load(f)
                found_default = False
                for speaker, path in data.items():
                    if speaker == "default":
                        found_default = True
                    path = validate_model_arg(path, "dictionary")
                if not found_default:
                    raise NoDefaultSpeakerDictionaryError()
    else:
        if os.path.splitext(name)[1]:
            raise ModelExtensionError(name, model_type, model_class.extensions)
        else:
            raise PretrainedModelNotFoundError(name, model_type, available_models)
    return name
