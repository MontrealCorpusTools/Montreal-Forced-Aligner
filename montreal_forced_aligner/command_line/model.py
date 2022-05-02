"""Command line functions for interacting with MFA models"""
from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, Optional

from montreal_forced_aligner.config import get_temporary_directory
from montreal_forced_aligner.data import PhoneSetType
from montreal_forced_aligner.exceptions import (
    FileArgumentNotFoundError,
    ModelLoadError,
    ModelTypeNotSupportedError,
    MultipleModelTypesFoundError,
    PretrainedModelNotFoundError,
    RemoteModelNotFoundError,
)
from montreal_forced_aligner.models import MODEL_TYPES, Archive, ModelManager
from montreal_forced_aligner.utils import guess_model_type

if TYPE_CHECKING:
    from argparse import Namespace


__all__ = [
    "inspect_model",
    "validate_args",
    "save_model",
    "run_model",
]


def inspect_model(path: str) -> None:
    """
    Inspect a model and print out metadata information about it

    Parameters
    ----------
    path: str
        Path to model
    """
    working_dir = os.path.join(get_temporary_directory(), "models", "inspect")
    ext = os.path.splitext(path)[1]
    model = None
    if ext == Archive.extensions[0]:  # Figure out what kind of model it is
        a = Archive(path, working_dir)
        model = a.get_subclass_object()
    else:
        for model_class in MODEL_TYPES.values():
            if model_class.valid_extension(path):
                if model_class == MODEL_TYPES["dictionary"]:
                    model = model_class(path, working_dir, phone_set_type=PhoneSetType.AUTO)
                else:
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
    model_class = MODEL_TYPES[model_type]
    if output_name:
        out_path = model_class.get_pretrained_path(output_name, enforce_existence=False)
    else:
        out_path = model_class.get_pretrained_path(model_name, enforce_existence=False)
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
        if args.name:
            manager = ModelManager()
            manager.refresh_remote()
            available_languages = manager.remote_models[args.model_type]
            if args.name not in available_languages:
                raise RemoteModelNotFoundError(
                    args.name, args.model_type, list(available_languages.keys())
                )
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
                model_class = MODEL_TYPES[args.model_type]
                path = model_class.get_pretrained_path(args.name)
                if path is None:
                    raise PretrainedModelNotFoundError(
                        args.name, args.model_type, model_class.get_available_models()
                    )
            else:
                found_model_types = []
                path = None
                for model_type, model_class in MODEL_TYPES.items():
                    p = model_class.get_pretrained_path(args.name)
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
    """
    validate_args(args)
    manager = ModelManager(token=args.github_token)
    if args.action == "download" and args.name:
        manager.download_model(args.model_type, args.name, args.ignore_cache)
    elif args.action == "download":
        manager.print_remote_models(args.model_type)
    elif args.action == "list":
        manager.print_local_models(args.model_type)
    elif args.action == "inspect":
        inspect_model(args.name)
    elif args.action == "save":
        save_model(args.path, args.model_type, args.name)
