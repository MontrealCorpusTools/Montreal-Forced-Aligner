"""Command line functions for interacting with MFA models"""
from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, List, Optional, Union

import requests

from montreal_forced_aligner.config import TEMP_DIR
from montreal_forced_aligner.exceptions import (
    FileArgumentNotFoundError,
    ModelLoadError,
    ModelTypeNotSupportedError,
    MultipleModelTypesFoundError,
    PretrainedModelNotFoundError,
)
from montreal_forced_aligner.helper import TerminalPrinter
from montreal_forced_aligner.models import MODEL_TYPES, Archive
from montreal_forced_aligner.utils import (
    get_available_models,
    get_pretrained_path,
    guess_model_type,
)

if TYPE_CHECKING:
    from argparse import Namespace


__all__ = [
    "list_downloadable_models",
    "download_model",
    "list_model",
    "inspect_model",
    "validate_args",
    "save_model",
    "run_model",
]


def list_downloadable_models(model_type: str) -> List[str]:
    """
    Generate a list of models available for download

    Parameters
    ----------
    model_type: str
        Model type to look up

    Returns
    -------
    List[str]
        Names of models
    """
    url = f"https://raw.githubusercontent.com/MontrealCorpusTools/mfa-models/main/{model_type}/index.txt"
    r = requests.get(url)
    if r.status_code == 404:
        raise Exception('Could not find model type "{}"'.format(model_type))
    out = r.text
    return out.split("\n")


def download_model(model_type: str, name: str) -> None:
    """
    Download a model to MFA's temporary directory

    Parameters
    ----------
    model_type: str
        Model type
    name: str
        Name of model
    """
    if name is None:
        downloadable = "\n".join(f"  - {x}" for x in list_downloadable_models(model_type))
        print(f"Available models to download for {model_type}:\n\n{downloadable}")
    try:
        mc = MODEL_TYPES[model_type]
        extension = mc.extensions[0]
        out_path = get_pretrained_path(model_type, name, enforce_existence=False)
    except KeyError:
        raise NotImplementedError(
            f"{model_type} models are not currently supported for downloading"
        )
    url = f"https://github.com/MontrealCorpusTools/mfa-models/raw/main/{model_type}/{name}{extension}"

    r = requests.get(url)
    with open(out_path, "wb") as f:
        f.write(r.content)


def list_model(model_type: Union[str, None]) -> None:
    """
    List all local pretrained models

    Parameters
    ----------
    model_type: str, optional
        Model type, will list models of all model types if None
    """
    printer = TerminalPrinter()
    if model_type is None:
        printer.print_information_line("Available models for use", "", level=0)
        for mt in MODEL_TYPES:
            names = get_available_models(mt)
            if names:
                printer.print_information_line(mt, names, value_color="green")
            else:
                printer.print_information_line(mt, "No models found", value_color="yellow")
    else:
        printer.print_information_line(f"Available models for use {model_type}", "", level=0)
        names = get_available_models(model_type)
        if names:
            for name in names:

                printer.print_information_line("", name, value_color="green", level=1)
        else:
            printer.print_information_line("", "No models found", value_color="yellow", level=1)


def inspect_model(path: str) -> None:
    """
    Inspect a model and print out metadata information about it

    Parameters
    ----------
    path: str
        Path to model
    """
    working_dir = os.path.join(TEMP_DIR, "models", "inspect")
    ext = os.path.splitext(path)[1]
    model = None
    if ext == Archive.extensions[0]:  # Figure out what kind of model it is
        a = Archive(path, working_dir)
        model = a.get_subclass_object()
    else:
        for model_class in MODEL_TYPES.values():
            if model_class.valid_extension(path):
                model = model_class(path, working_dir)
    if not model:
        raise ModelLoadError(path)
    model.pretty_print()


def save_model(path: str, model_type: str, output_name: Optional[str]) -> None:
    """
    Save a model to pretrained folder for later use

    Parameters
    ----------
    path: str
        Path to model
    model_type: str
        Type of model
    """
    model_name = os.path.splitext(os.path.basename(path))[0]
    if output_name:
        out_path = get_pretrained_path(model_type, output_name, enforce_existence=False)
    else:
        out_path = get_pretrained_path(model_type, model_name, enforce_existence=False)
    shutil.copyfile(path, out_path)


def validate_args(args: Namespace) -> None:
    """
    Validate the command line arguments

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Parsed command line arguments

    Raises
    ------
    :class:`~montreal_forced_aligner.exceptions.ArgumentError`
        If there is a problem with any arguments
    :class:`~montreal_forced_aligner.exceptions.ModelTypeNotSupportedError`
        If the type of model is not supported
    :class:`~montreal_forced_aligner.exceptions.FileArgumentNotFoundError`
        If the file specified is not found
    :class:`~montreal_forced_aligner.exceptions.PretrainedModelNotFoundError`
        If the pretrained model specified is not found
    :class:`~montreal_forced_aligner.exceptions.ModelExtensionError`
        If the extension is not valid for the specified model type
    :class:`~montreal_forced_aligner.exceptions.MultipleModelTypesFoundError`
        If multiple model types match the name
    """
    if args.action == "download":
        if args.model_type not in MODEL_TYPES:
            raise ModelTypeNotSupportedError(args.model_type, MODEL_TYPES)
        elif args.model_type:
            args.model_type = args.model_type.lower()
        if args.name is not None:
            available_languages = list_downloadable_models(args.model_type)
            if args.name not in available_languages:
                raise PretrainedModelNotFoundError(args.name, args.model_type, available_languages)
    elif args.action == "list":
        if args.model_type and args.model_type.lower() not in MODEL_TYPES:
            raise ModelTypeNotSupportedError(args.model_type, MODEL_TYPES)
        elif args.model_type:
            args.model_type = args.model_type.lower()
    elif args.action == "inspect":
        if args.model_type and args.model_type not in MODEL_TYPES:
            raise ModelTypeNotSupportedError(args.model_type, MODEL_TYPES)
        elif args.model_type:
            args.model_type = args.model_type.lower()
        possible_model_types = guess_model_type(args.name)
        if not possible_model_types:
            if args.model_type:
                path = get_pretrained_path(args.model_type, args.name)
                if path is None:
                    raise PretrainedModelNotFoundError(
                        args.name, args.model_type, get_available_models(args.model_type)
                    )
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
    elif args.action == "save":
        if not os.path.exists(args.path):
            raise FileArgumentNotFoundError(args.path)


def run_model(args: Namespace) -> None:
    """
    Wrapper function for running model utility commands

    Parameters
    ----------
    args: :class:`~argparse.Namespace`
        Parsed command line arguments
    unknown: List[str]
        Parsed command line arguments to be passed to the configuration objects
    """
    validate_args(args)
    if args.action == "download":
        download_model(args.model_type, args.name)
    elif args.action == "list":
        list_model(args.model_type)
    elif args.action == "inspect":
        inspect_model(args.name)
    elif args.action == "save":
        save_model(args.path, args.model_type, args.name)
