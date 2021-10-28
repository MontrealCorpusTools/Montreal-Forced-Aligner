from __future__ import annotations
from typing import TYPE_CHECKING, List, Union
if TYPE_CHECKING:
    from argparse import Namespace
import requests
import os
from montreal_forced_aligner.exceptions import FileArgumentNotFoundError, PretrainedModelNotFoundError, \
    ModelLoadError, MultipleModelTypesFoundError, ModelTypeNotSupportedError
from montreal_forced_aligner.config import TEMP_DIR
from montreal_forced_aligner.models import MODEL_TYPES, Archive
from montreal_forced_aligner.utils import get_available_models, get_pretrained_path, guess_model_type
from montreal_forced_aligner.helper import TerminalPrinter


def list_downloadable_languages(model_type: str) -> List[str]:
    url = f'https://raw.githubusercontent.com/MontrealCorpusTools/mfa-models/main/{model_type}/index.txt'
    r = requests.get(url)
    if r.status_code == 404:
        raise Exception('Could not find model type "{}"'.format(model_type))
    out = r.text
    return out.split('\n')


def download_model(model_type: str, name: str) -> None:
    if name is None:
        downloadable = "\n".join(f'  - {x}' for x in list_downloadable_languages(model_type))
        print(f'Available models to download for {model_type}:\n\n{downloadable}')
    try:
        mc = MODEL_TYPES[model_type]
        extension = mc.extensions[0]
        out_path = get_pretrained_path(model_type, name)
    except KeyError:
        raise NotImplementedError(f"{model_type} models are not currently supported for downloading")
    url = f'https://github.com/MontrealCorpusTools/mfa-models/raw/main/{model_type}/{name}{extension}'

    r = requests.get(url)
    with open(out_path, 'wb') as f:
        f.write(r.content)


def list_model(model_type: Union[str, None]) -> None:
    printer = TerminalPrinter()
    if model_type is None:
        printer.print_information_line('Available models for use', '', level=0)
        for mt in MODEL_TYPES:
            names = get_available_models(mt)
            if names:
                printer.print_information_line(mt, names, value_color='green')
            else:
                printer.print_information_line(mt, 'No models found', value_color='yellow')
    else:
        printer.print_information_line(f'Available models for use {model_type}', '', level=0)
        names = get_available_models(model_type)
        if names:
            for name in names:

                printer.print_information_line('', name, value_color='green', level=1)
        else:
            printer.print_information_line('', 'No models found', value_color='yellow', level=1)

def inspect_model(path: str) -> None:
    working_dir = os.path.join(TEMP_DIR, 'models', 'inspect')
    ext = os.path.splitext(path)[1]
    model = None
    if ext == Archive.extensions[0]: # Figure out what kind of model it is
        a = Archive(path, working_dir)
        model = a.get_subclass_object()
    else:
        for model_type, model_class in MODEL_TYPES.items():
            if model_class.valid_extension(path):
                model = model_class(path, working_dir)
    if not model:
        raise ModelLoadError(path)
    model.pretty_print()

def validate_args(args: Namespace) -> None:
    if args.action == 'download':
        if args.model_type not in MODEL_TYPES:
            raise ModelTypeNotSupportedError(args.model_type, MODEL_TYPES)
        elif args.model_type:
            args.model_type = args.model_type.lower()
        if args.name is not None:
            available_languages = list_downloadable_languages(args.model_type)
            if args.name not in available_languages:
                possible = ', '.join(available_languages)
                raise PretrainedModelNotFoundError(args.name, args.model_type, possible)
    elif args.action == 'list':
        if args.model_type and args.model_type.lower() not in MODEL_TYPES:
            raise ModelTypeNotSupportedError(args.model_type, MODEL_TYPES)
        elif args.model_type:
            args.model_type = args.model_type.lower()
    elif args.action == 'inspect':
        if args.model_type and args.model_type not in MODEL_TYPES:
            raise ModelTypeNotSupportedError(args.model_type, MODEL_TYPES)
        elif args.model_type:
            args.model_type = args.model_type.lower()
        possible_model_types = guess_model_type(args.name)
        if not possible_model_types:
            if args.model_type:
                path = get_pretrained_path(args.model_type, args.name)
                if path is None:
                    raise PretrainedModelNotFoundError(args.name, args.model_type, get_available_models(args.model_type))
            else:
                found_model_types = []
                path = None
                for model_type in MODEL_TYPES:
                    p = get_pretrained_path(model_type, args.name)
                    if p is not None:
                        path = p
                        found_model_types.append(model_type)
                if len(found_model_types) > 1:
                    raise MultipleModelTypesFoundError(args.name, found_model_types)
                if path is None:
                    raise PretrainedModelNotFoundError(args.name)
            args.name = path
        else:
            if not os.path.exists(args.name):
                raise FileArgumentNotFoundError(args.name)
    elif args.action == 'save':
        if not os.path.exists(args.path):
            raise FileArgumentNotFoundError(args.path)


def run_model(args: Namespace) -> None:
    validate_args(args)
    if args.action == 'download':
        download_model(args.model_type, args.name)
    elif args.action == 'list':
        list_model(args.model_type)
    elif args.action == 'inspect':
        inspect_model(args.name)
    elif args.action == 'save':
        save_model(args.path, args.name)