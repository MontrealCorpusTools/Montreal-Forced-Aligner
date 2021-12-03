"""
Exception classes
=================

"""
from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Collection, Optional

from colorama import Fore, Style

from montreal_forced_aligner.helper import comma_join

if TYPE_CHECKING:
    from montreal_forced_aligner.dictionary.pronunciation import PronunciationDictionaryMixin
    from montreal_forced_aligner.models import G2PModel
    from montreal_forced_aligner.textgrid import CtmInterval


__all__ = [
    "MFAError",
    "SoxError",
    "G2PError",
    "ConfigError",
    "LMError",
    "LanguageModelNotFoundError",
    "ModelExtensionError",
    "ThirdpartyError",
    "TrainerError",
    "ModelError",
    "CorpusError",
    "ModelLoadError",
    "CorpusReadError",
    "ArgumentError",
    "AlignerError",
    "AlignmentError",
    "AlignmentExportError",
    "NoSuccessfulAlignments",
    "KaldiProcessingError",
    "TextParseError",
    "TextGridParseError",
    "DictionaryError",
    "NoDefaultSpeakerDictionaryError",
    "DictionaryPathError",
    "DictionaryFileError",
    "FileArgumentNotFoundError",
    "PretrainedModelNotFoundError",
    "MultipleModelTypesFoundError",
    "ModelTypeNotSupportedError",
    "PronunciationAcousticMismatchError",
    "PronunciationOrthographyMismatchError",
]


class MFAError(Exception):
    """
    Base exception class
    """

    def __init__(self, *args, **kwargs):
        from .config import USE_COLORS

        self.red_text = ""
        self.bright_text = ""
        self.green_text = ""
        self.reset_text = ""
        if USE_COLORS:
            self.red_text = Fore.RED
            self.bright_text = Style.BRIGHT
            self.green_text = Fore.GREEN
            self.reset_text = Style.RESET_ALL
        self.message = ""
        if args and isinstance(args[0], str):
            self.message = args[0]

    def error_text(self, text: str) -> str:
        """
        Highlight text as an error

        Parameters
        ----------
        text: str
            Text to highlight

        Returns
        -------
        str
            Highlighted text
        """
        return f"{self.red_text}{text}{self.reset_text}"

    def emphasized_text(self, text: str) -> str:
        """
        Highlight text as emphasis

        Parameters
        ----------
        text: str
            Text to highlight

        Returns
        -------
        str
            Highlighted text
        """
        return f"{self.bright_text}{text}{self.reset_text}"

    def pass_text(self, text: str) -> str:
        """
        Highlight text as good

        Parameters
        ----------
        text: str
            Text to highlight

        Returns
        -------
        str
            Highlighted text
        """
        return f"{self.green_text}{text}{self.reset_text}"

    def __str__(self) -> str:
        """Output the error"""
        return f"{self.error_text(type(self).__name__)}: {self.message}"


class PlatformError(MFAError):
    """
    Exception class for platform compatibility issues

    Parameters
    ----------
    functionality_name: str
        Functionality not available on the current platform
    """

    def __init__(self, functionality_name):
        super().__init__()
        self.message = f"Functionality for {self.emphasized_text(functionality_name)} is not available on {self.error_text(sys.platform)}."
        if sys.platform == "win32":
            self.message += (
                f" If you'd like to use {self.emphasized_text(functionality_name)} on Windows, please follow the MFA installation "
                f"instructions for the Windows Subsystem for Linux (WSL)."
            )


class ThirdpartyError(MFAError):
    """
    Exception class for errors in third party binary (usually Kaldi or OpenFst)

    Parameters
    ----------
    binary_name: str
        Name of third party binary
    open_fst: bool, optional
        Flag for the error having to do with OpenFst
    open_blas: bool, optional
        Flag for the error having to do with the BLAS library
    libc: bool, optional
        Flag for the error having to do with the system libraries
    sox: bool, optional
        Flag for the error having to do with SoX
    """

    def __init__(self, binary_name, open_fst=False, open_blas=False, libc=False, sox=False):
        super().__init__()
        self.message = f"Could not find '{self.error_text(binary_name)}'. "
        extra = "Please ensure that you have downloaded the correct binaries."
        if open_fst:
            extra = (
                f"Please ensure that you are in an environment that has the {self.emphasized_text('openfst')} conda package installed, "
                f"or that the {self.emphasized_text('openfst')} binaries are on your path if you compiled them yourself."
            )
        elif open_blas:
            extra = f"Try installing {self.emphasized_text('openblas')} via system package manager or verify it's on your system path?"
        elif libc:
            extra = (
                f"You likely have a different version of {self.emphasized_text('glibc')} than the precompiled binaries use. "
                f"Try compiling {self.emphasized_text('Kaldi')} on your machine and collecting the binaries via the "
                f"`{self.pass_text('mfa thirdparty kaldi')}` command."
            )
        elif sox:
            self.message = ""
            extra = (
                f"Your version of {self.emphasized_text('sox')} does not support the file format in your corpus. "
                f"Try installing another version of {self.emphasized_text('sox')} with support for {self.error_text(binary_name)}."
            )
        self.message += extra


# Model Errors


class ModelError(MFAError):
    """
    Exception class related to MFA model archives
    """

    pass


class ModelLoadError(ModelError):
    """
    Exception during loading of a model archive

    Parameters
    ----------
    path: str
        Path of the model archive
    """

    def __init__(self, path: str):
        super(ModelLoadError, self).__init__()
        self.message = f"The archive {self.error_text(path)} could not be parsed as an MFA model"


# Dictionary Errors


class DictionaryError(MFAError):
    """
    Exception class for errors in creating dictionary objects
    """

    pass


class NoDefaultSpeakerDictionaryError(DictionaryError):
    """
    Exception class for errors in creating MultispeakerDictionary objects
    """

    def __init__(self):
        super().__init__()
        self.message = f'No "{self.error_text("default")}" dictionary was found.'


class DictionaryPathError(DictionaryError):
    """
    Exception class for errors in locating paths for dictionary objects

    Parameters
    ----------
    input_path: str
        Path of the pronunciation dictionary
    """

    def __init__(self, input_path: str):
        super().__init__()
        self.message = (
            f"The specified path for the dictionary ({self.error_text(input_path)}) was not found."
        )


class DictionaryFileError(DictionaryError):
    """
    Exception class for file type being wrong for DictionaryModel objects

    Parameters
    ----------
    input_path: str
        Path of the pronunciation dictionary
    """

    def __init__(self, input_path: str):
        super().__init__()
        self.message = (
            f"The specified path for the dictionary ({self.error_text(input_path)}) is not a file."
        )


# Corpus Errors


class CorpusError(MFAError):
    """
    Class for errors in creating Corpus objects
    """

    pass


class CorpusReadError(CorpusError):
    """
    Class for errors in reading a file

    Parameters
    ----------
    file_name: str
        File name that was not readable
    """

    def __init__(self, file_name: str):
        MFAError.__init__(self)
        self.file_name = file_name


class TextParseError(CorpusReadError):
    """
    Class for errors parsing lab and txt files

    Parameters
    ----------
    file_name: str
        File name that had the error
    """

    def __init__(self, file_name: str):
        MFAError.__init__(self)
        self.file_name = file_name


class TextGridParseError(CorpusReadError):
    """
    Class capturing TextGrid reading errors

    Parameters
    ----------
    file_name: str
        File name that had the error
    error: str
        Error in TextGrid file
    """

    def __init__(self, file_name: str, error: str):
        MFAError.__init__(self)
        self.file_name = file_name
        self.error = error
        self.message = f"Reading {self.emphasized_text(self.file_name)} has the following error:\n\n{self.error}"


class SoxError(CorpusReadError):
    """
    Class for errors in calling and finding Sox
    """

    pass


# Aligner Errors


class AlignerError(MFAError):
    """
    Class for errors during alignment
    """

    pass


class AlignmentError(MFAError):
    """
    Class for errors during alignment

    Parameters
    ----------
    error_logs: list[str]
        List of Kaldi log files with errors
    """

    def __init__(self, error_logs: list[str]):
        super().__init__()
        output = "\n".join(error_logs)
        self.message = (
            f"There were {len(error_logs)} job(s) with errors.  "
            f"For more information, please see the following logs:\n\n{output}"
        )


class AlignmentExportError(AlignmentError):
    """
    Class for errors in exporting alignments

    Parameters
    ----------
    error_dict: dict[tuple[str, int], str]
        Error dictionary mapping export stage and job to the error encountered

    """

    def __init__(self, error_dict: dict[tuple[str, int], str]):
        MFAError.__init__(self)

        message = "Error was encountered in processing CTMs:\n\n"
        for key, error in error_dict.items():
            message += f"{key}:\n{error}"
        self.message = message


class CtmError(AlignmentError):
    """
    Class for errors in creating CTM intervals

    Parameters
    ----------
    ctm: CtmInterval
        CTM interval that was not parsed correctly

    """

    def __init__(self, ctm: CtmInterval):
        MFAError.__init__(self)

        self.message = f"Error was encountered in processing CTM interval: {ctm}"


class NoSuccessfulAlignments(AlignerError):
    """
    Class for errors where nothing could be aligned
    """

    pass


class PronunciationAcousticMismatchError(AlignerError):
    """
    Exception class for when an acoustic model and pronunciation dictionary have different phone sets

    Parameters
    ----------
    missing_phones: Collection[str]
        Phones that are not in the acoustic model
    """

    def __init__(self, missing_phones: Collection[str]):
        super().__init__()
        missing_phones = [f"{self.error_text(x)}" for x in sorted(missing_phones)]
        self.message = (
            f"There were phones in the dictionary that do not have acoustic models: "
            f"{comma_join(missing_phones)}"
        )


class PronunciationOrthographyMismatchError(AlignerError):
    """
    Exception class for missing graphemes in a G2P model

    Parameters
    ----------
    g2p_model: :class:`~montreal_forced_aligner.models.G2PModel`
        Specified G2P model
    dictionary: :class:`~montreal_forced_aligner.dictionary.pronunciation.PronunciationDictionaryMixin`
        Specified dictionary
    """

    def __init__(self, g2p_model: G2PModel, dictionary: PronunciationDictionaryMixin):
        super().__init__()
        missing_graphs = dictionary.graphemes - set(g2p_model.meta["graphemes"])
        missing_graphs = [f"{self.error_text(x)}" for x in sorted(missing_graphs)]
        self.message = (
            f"There were graphemes in the corpus that are not covered by the G2P model: "
            f"{comma_join(missing_graphs)}"
        )


# Command line exceptions


class ArgumentError(MFAError):
    """
    Exception class for errors parsing command line arguments
    """

    pass


class FileArgumentNotFoundError(ArgumentError):
    """
    Exception class for not finding a specified file

    Parameters
    ----------
    path: str
        Path not found
    """

    def __init__(self, path):
        super().__init__()
        self.message = f'Could not find "{self.error_text(path)}".'


class PretrainedModelNotFoundError(ArgumentError):
    """
    Exception class for not finding a specified pretrained model

    Parameters
    ----------
    name: str
        Model name
    model_type: str, optional
        Model type searched
    available: list[str], optional
        List of models that were found
    """

    def __init__(
        self, name: str, model_type: Optional[str] = None, available: Optional[list[str]] = None
    ):
        super().__init__()
        extra = ""
        if model_type:
            extra += f" for {model_type}"
        message = f'Could not find a model named "{self.error_text(name)}"{extra}.'
        if available:
            available = [f"{self.pass_text(x)}" for x in available]
            message += f" Available: {comma_join(available)}."
        self.message = message


class MultipleModelTypesFoundError(ArgumentError):
    """
    Exception class for finding multiple model types that could map to a given name

    Parameters
    ----------
    name: str
        Model name
    possible_model_types: list[str]
        List of model types that have a model with the given name
    """

    def __init__(self, name: str, possible_model_types: list[str]):
        super().__init__()

        possible_model_types = [f"{self.error_text(x)}" for x in possible_model_types]
        self.message = (
            f'Found multiple model types for "{self.error_text(name)}": {", ".join(possible_model_types)}. '
            f"Please specify a model type to inspect."
        )


class ModelExtensionError(ArgumentError):
    """
    Exception class for a model not having the correct extension

    Parameters
    ----------
    name: str
        Model name
    model_type: str
        Model type
    extensions: list[str]
        Extensions that the model supports
    """

    def __init__(self, name: str, model_type: str, extensions: list[str]):
        super().__init__()
        extra = ""
        if model_type:
            extra += f" for {model_type}"
        message = (
            f'The path "{self.error_text(name)}" does not have the correct extensions{extra}.'
        )
        if extensions:
            available = [f"{self.pass_text(x)}" for x in extensions]
            message += f" Possible extensions: {comma_join(available)}."
        self.message = message


class ModelTypeNotSupportedError(ArgumentError):
    """
    Exception class for a model type not being supported

    Parameters
    ----------
    model_type: str
        Model type
    model_types: list[str]
        List of supported model types
    """

    def __init__(self, model_type, model_types):
        super().__init__()
        message = f'The model type "{self.error_text(model_type)}" is not supported.'
        if model_types:
            model_types = [f"{self.pass_text(x)}" for x in sorted(model_types)]
            message += f" Possible model types: {comma_join(model_types)}."
        self.message = message


class ConfigError(MFAError):
    """
    Exception class for errors in configuration
    """

    pass


class RootDirectoryError(ConfigError):
    """
    Exception class for errors using the MFA_ROOT_DIR
    """

    def __init__(self, temporary_directory, variable):
        super().__init__()
        self.message = (
            f"Could not create a root MFA temporary directory (tried {self.error_text(temporary_directory)}), "
            f"please specify a write-able directory via the {self.emphasized_text(variable)} environment variable."
        )


class TrainerError(MFAError):
    """
    Exception class for errors in trainers
    """

    pass


class G2PError(MFAError):
    """
    Exception class for errors in G2P
    """

    pass


class LMError(MFAError):
    """
    Exception class for errors in language models
    """

    pass


class LanguageModelNotFoundError(LMError):
    """
    Exception class for a language model not being found
    """

    def __init__(self):
        super(LanguageModelNotFoundError, self).__init__()
        self.message = "Could not find a suitable language model"


class KaldiProcessingError(MFAError):
    """
    Exception class for when a Kaldi binary has an exception

    Parameters
    ----------
    error_logs: list[str]
        List of Kaldi logs that had errors
    log_file: str, optional
        Overall log file to find more information
    """

    def __init__(self, error_logs: list[str], log_file: Optional[str] = None):
        super().__init__()
        self.message = (
            f"There were {len(error_logs)} job(s) with errors when running Kaldi binaries."
        )
        if log_file is not None:
            self.message += f" For more details, please check {self.error_text(log_file)}"
        self.error_logs = error_logs
        self.log_file = log_file

    def update_log_file(self, logger: logging.Logger) -> None:
        """
        Update the log file output

        Parameters
        ----------
        logger: logging.Logger
            Logger
        """
        if logger.handlers:
            self.log_file = logger.handlers[0].baseFilename
        self.message = (
            f"There were {len(self.error_logs)} job(s) with errors when running Kaldi binaries."
        )
        if self.log_file is not None:
            self.message += f" For more details, please check {self.error_text(self.log_file)}"
