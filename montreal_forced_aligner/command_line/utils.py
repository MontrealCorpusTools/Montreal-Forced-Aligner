from __future__ import annotations
import os
import yaml
from ..exceptions import FileArgumentNotFoundError, PretrainedModelNotFoundError, ModelExtensionError,\
    NoDefaultSpeakerDictionaryError, ModelTypeNotSupportedError

from ..utils import get_pretrained_path, get_available_models

from ..models import MODEL_TYPES


def validate_model_arg(name: str, model_type: str) -> str:
    if model_type not in MODEL_TYPES:
        raise ModelTypeNotSupportedError(model_type, MODEL_TYPES)
    available_models = get_available_models(model_type)
    model_class = MODEL_TYPES[model_type]
    if name in available_models:
        name = get_pretrained_path(model_type, name)
    elif model_class.valid_extension(name):
        if not os.path.exists(name):
            raise FileArgumentNotFoundError(name)
        if model_type == 'dictionary' and os.path.splitext(name)[1].lower() == '.yaml':
            with open(name, 'r', encoding='utf8') as f:
                data = yaml.safe_load(f)
                found_default = False
                for speaker, path in data.items():
                    if speaker == 'default':
                        found_default = True
                    path = validate_model_arg(path, 'dictionary')
                if not found_default:
                    raise NoDefaultSpeakerDictionaryError()
    else:
        if os.path.splitext(name)[1]:
            raise ModelExtensionError(name, model_type, model_class.extensions)
        else:
            raise PretrainedModelNotFoundError(name, model_type, available_models)
    return name


