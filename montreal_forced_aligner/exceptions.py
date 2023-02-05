"""
Exception classes
=================

"""
from __future__ import annotations

import datetime
import logging
import re
import sys
import typing
from typing import TYPE_CHECKING, Collection, Dict, List, Optional

import requests.structures

from montreal_forced_aligner.helper import TerminalPrinter, comma_join

if TYPE_CHECKING:
    from montreal_forced_aligner.dictionary.mixins import DictionaryMixin
    from montreal_forced_aligner.models import G2PModel
    from montreal_forced_aligner.textgrid import CtmInterval


__all__ = [
    "MFAError",
    "SoxError",
    "G2PError",
    "CtmError",
    "PyniniAlignmentError",
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

    def __init__(self, base_error_message: str, *args, **kwargs):
        self.printer = TerminalPrinter()
        self.message_lines: List[str] = [base_error_message]

    @property
    def message(self) -> str:
        """Formatted exception message"""
        return "\n".join(self.printer.format_info_lines(self.message_lines))

    def __str__(self) -> str:
        """Output the error"""
        message = self.printer.error_text(type(self).__name__) + ":"
        self.printer.indent_level += 1
        message += "\n\n" + self.message
        self.printer.indent_level -= 1
        return message


class PlatformError(MFAError):
    """
    Exception class for platform compatibility issues

    Parameters
    ----------
    functionality_name: str
        Functionality not available on the current platform
    """

    def __init__(self, functionality_name):
        super().__init__("")
        self.message_lines = [
            f"Functionality for {self.printer.emphasized_text(functionality_name)} is not available on {self.printer.error_text(sys.platform)}."
        ]
        if sys.platform == "win32":
            self.message_lines.append("")
            self.message_lines.append(
                f" If you'd like to use {self.printer.emphasized_text(functionality_name)} on Windows, please follow the MFA installation "
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

    def __init__(
        self, binary_name, open_fst=False, open_blas=False, libc=False, sox=False, error_text=None
    ):
        super().__init__("")
        if error_text:
            self.message_lines = [
                f"There was an error when invoking '{self.printer.error_text(binary_name)}':",
                error_text,
                "This likely indicates that MFA's dependencies were not correctly installed, or there is an issue with your Conda environment.",
                "If you are in the correct environment, please try re-creating the environment from scratch as a first step, i.e.:",
                self.printer.pass_text(
                    "conda create -n aligner -c conda-forge montreal-forced-aligner"
                ),
            ]
        else:
            self.message_lines = [f"Could not find '{self.printer.error_text(binary_name)}'."]
            self.message_lines.append(
                "Please ensure that you have installed MFA's conda dependencies and are in the correct environment."
            )
            if open_fst:
                self.message_lines.append(
                    f"Please ensure that you are in an environment that has the {self.printer.emphasized_text('openfst')} conda package installed, "
                    f"or that the {self.printer.emphasized_text('openfst')} binaries are on your path if you compiled them yourself."
                )
            elif open_blas:
                self.message_lines.append(
                    f"Try installing {self.printer.emphasized_text('openblas')} via system package manager or verify it's on your system path?"
                )
            elif libc:
                self.message_lines.append(
                    f"You likely have a different version of {self.printer.emphasized_text('glibc')} than the packages binaries use. "
                    f"Try compiling {self.printer.emphasized_text('Kaldi')} on your machine and collecting the binaries via the "
                    f"{self.printer.pass_text('mfa thirdparty kaldi')} command."
                )
            elif sox:
                self.message_lines = []
                self.message_lines.append(
                    f"Your version of {self.printer.emphasized_text('sox')} does not support the file format in your corpus. "
                    f"Try installing another version of {self.printer.emphasized_text('sox')} with support for {self.printer.error_text(binary_name)}."
                )


# Feature Generation Errors


class FeatureGenerationError(MFAError):
    """
    Exception class related to generating features
    """

    pass


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
        super().__init__("")
        self.message_lines = [
            f"The archive {self.printer.error_text(path)} could not be parsed as an MFA model."
        ]


class ModelSaveError(ModelError):
    """
    Exception during saving of a model archive

    Parameters
    ----------
    path: str
        Path of the model archive
    """

    def __init__(self, path: str):
        super().__init__("")
        self.message_lines = [
            f"The archive {self.printer.error_text(path)} already exists.",
            "Please specify --overwrite if you would like to overwrite  it.",
        ]


class ModelsConnectionError(ModelError):
    """
    Exception during connecting to online repo for downloading models

    Parameters
    ----------
    response_code: int
        Response code for the request
    response: dict[str, Any]
        Response dictionary
    headers: requests.structures.CaseInsensitiveDict
        Request headers
    """

    def __init__(
        self,
        response_code: int,
        response: typing.Dict[str, typing.Any],
        headers: requests.structures.CaseInsensitiveDict,
    ):
        super().__init__("")
        if response_code == 403 and "API rate limit" in response["message"]:
            rate_limit = headers["x-ratelimit-limit"]
            rate_limit_reset = datetime.datetime.fromtimestamp(int(headers["x-ratelimit-reset"]))
            self.message_lines = [
                f"Current hourly rate limit ({self.printer.error_text(rate_limit)} per hour) has been exceeded for the GitHub API.",
                "You can increase it by providing a personal authentication token to via --github_token.",
                f"The rate limit will reset at {self.printer.pass_text(rate_limit_reset)}",
            ]
        else:
            self.message_lines = [
                f"The response returned code {response_code}:  {response['message']}"
            ]


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
        super().__init__("")
        self.message_lines = [f'No "{self.printer.error_text("default")}" dictionary was found.']


class DictionaryPathError(DictionaryError):
    """
    Exception class for errors in locating paths for dictionary objects

    Parameters
    ----------
    input_path: str
        Path of the pronunciation dictionary
    """

    def __init__(self, input_path: str):
        super().__init__("")
        self.message_lines = [
            f"The specified path for the dictionary ({self.printer.error_text(input_path)}) was not found."
        ]


class DictionaryFileError(DictionaryError):
    """
    Exception class for file type being wrong for DictionaryModel objects

    Parameters
    ----------
    input_path: str
        Path of the pronunciation dictionary
    """

    def __init__(self, input_path: str):
        super().__init__("")
        self.message_lines = [
            f"The specified path for the dictionary ({self.printer.error_text(input_path)}) is not a file."
        ]


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
        super().__init__("")
        self.message_lines = [f"There was an error reading {self.printer.error_text(file_name)}."]


class TextParseError(CorpusReadError):
    """
    Class for errors parsing lab and txt files

    Parameters
    ----------
    file_name: str
        File name that had the error
    """

    def __init__(self, file_name: str):
        super().__init__("")
        self.message_lines = [
            f"There was an error decoding {self.printer.error_text(file_name)}, "
            f"maybe try resaving it as utf8?"
        ]


class TextGridParseError(CorpusReadError):
    """
    Class capturing TextGrid reading errors

    Parameters
    ----------
    file_name: str
        File name
    error: str
        Error in TextGrid file
    """

    def __init__(self, file_name: str, error: str):
        super().__init__("")
        self.file_name = file_name
        self.error = error
        self.message_lines.extend(
            [
                f"Reading {self.printer.emphasized_text(file_name)} has the following error:",
                "",
                "",
                self.error,
            ]
        )


class SoxError(CorpusReadError):
    """
    Class for errors in calling and finding Sox
    """

    pass


class SoundFileError(CorpusReadError):
    """
    Class for errors in sound files

    Parameters
    ----------
    file_name: str
        File name
    error: str
        Error in TextGrid file
    """

    def __init__(self, file_name: str, error: str):
        super().__init__("")
        self.file_name = file_name
        self.error = error
        self.message_lines.extend(
            [
                f"Reading {self.printer.emphasized_text(file_name)} has the following error:",
                "",
                "",
                self.error,
            ]
        )


# Aligner Errors


class AlignerError(MFAError):
    """
    Class for errors during alignment
    """

    pass


class NoAlignmentsError(MFAError):
    """
    Class for errors during alignment
    """

    def __init__(self, num_utterances, beam_size, retry_beam_size):
        super(NoAlignmentsError, self).__init__(
            f"There were no successful alignments for {num_utterances} utterances."
        )
        self.message_lines.append(
            f"The current set up used a beam of {beam_size} and a retry beam of {retry_beam_size}."
        )
        suggested_beam_size = beam_size * 10
        suggested_retry_beam_size = suggested_beam_size * 4
        self.message_lines.append(
            f'You can try rerunning with a larger beam (i.e. "mfa align ... --beam {suggested_beam_size} --retry_beam {suggested_retry_beam_size}").'
        )
        self.message_lines.append(
            'If increasing the beam size does not help, then there are likely issues with the corpus, dictionary, or acoustic model, and can be further diagnosed with the "mfa validate" command'
        )


class AlignmentError(MFAError):
    """
    Class for errors during alignment

    Parameters
    ----------
    error_logs: list[str]
        List of Kaldi log files with errors
    """

    def __init__(self, error_logs: List[str]):
        super().__init__("")
        self.message_lines = [
            f"There were {self.printer.error_text(len(error_logs))} job(s) with errors. "
            f"For more information, please see:",
            "",
            "",
        ]
        for path in error_logs:
            self.message_lines.append(self.printer.error_text(path))


class AlignmentExportError(AlignmentError):
    """
    Class for errors in exporting alignments

    Parameters
    ----------
    error_dict: dict[tuple[str, int], str]
        Error dictionary mapping export stage and job to the error encountered

    """

    def __init__(self, path: str, error_lines: List[str]):
        MFAError.__init__(self, f"Error was encountered in exporting {path}:")
        self.path = path
        self.message_lines.append("")
        self.message_lines.append("")

        self.message_lines.extend(error_lines)


class CtmError(AlignmentError):
    """
    Class for errors in creating CTM intervals

    Parameters
    ----------
    ctm: :class:`~montreal_forced_aligner.data.CtmInterval`
        CTM interval that was not parsed correctly

    """

    def __init__(self, ctm: CtmInterval):
        MFAError.__init__(self, f"Error was encountered in processing CTM interval: {ctm}")


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
        super().__init__("There were phones in the dictionary that do not have acoustic models: ")
        missing_phones = [f"{self.printer.error_text(x)}" for x in sorted(missing_phones)]
        self.message_lines.append(comma_join(missing_phones))


class PronunciationOrthographyMismatchError(AlignerError):
    """
    Exception class for missing graphemes in a G2P model

    Parameters
    ----------
    g2p_model: :class:`~montreal_forced_aligner.models.G2PModel`
        Specified G2P model
    dictionary: :class:`~montreal_forced_aligner.dictionary.mixins.DictionaryMixin`
        Specified dictionary
    """

    def __init__(self, g2p_model: G2PModel, dictionary: DictionaryMixin):
        super().__init__(
            "There were graphemes in the corpus that are not covered by the G2P model:"
        )
        missing_graphs = dictionary.graphemes - set(g2p_model.meta["graphemes"])
        missing_graphs = [f"{self.printer.error_text(x)}" for x in sorted(missing_graphs)]
        self.message_lines.append(comma_join(missing_graphs))


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
        super().__init__("")
        self.message_lines = [f'Could not find "{self.printer.error_text(path)}".']


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
        self, name: str, model_type: Optional[str] = None, available: Optional[List[str]] = None
    ):
        super().__init__("")
        extra = ""
        if model_type:
            extra += f" for {model_type}"
        self.message_lines = [
            f'Could not find a model named "{self.printer.error_text(name)}"{extra}.'
        ]
        if available:
            available = [f"{self.printer.pass_text(x)}" for x in available]
            self.message_lines.append(f"Available: {comma_join(available)}.")


class RemoteModelNotFoundError(ArgumentError):
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
        self, name: str, model_type: Optional[str] = None, available: Optional[List[str]] = None
    ):
        super().__init__("")
        extra = ""
        if model_type:
            extra += f" for {model_type}"
        self.message_lines = [
            f'Could not find a model named "{self.printer.error_text(name)}"{extra}.'
        ]
        if available:
            available = [f"{self.printer.pass_text(x)}" for x in available]
            self.message_lines.append(f"Available: {comma_join(available)}.")
        self.message_lines.append(
            "You can see all available models either on https://mfa-models.readthedocs.io/en/latest/ or https://github.com/MontrealCorpusTools/mfa-models/releases."
        )
        if model_type:
            self.message_lines.append(
                f"If you're looking for a model from 1.0, please see https://github.com/MontrealCorpusTools/mfa-models/releases/tag/{model_type}-archive-v1.0."
            )


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

    def __init__(self, name: str, possible_model_types: List[str]):
        super().__init__("")
        self.message_lines = [f'Found multiple model types for "{self.printer.error_text(name)}":']
        possible_model_types = [f"{self.printer.error_text(x)}" for x in possible_model_types]
        self.message_lines.extend(
            [", ".join(possible_model_types), "Please specify a model type to inspect."]
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

    def __init__(self, name: str, model_type: str, extensions: List[str]):
        super().__init__("")
        extra = ""
        if model_type:
            extra += f" for {model_type}"
        self.message_lines = [
            f'The path "{self.printer.error_text(name)}" does not have the correct extensions{extra}.'
        ]

        if extensions:
            available = [f"{self.printer.pass_text(x)}" for x in extensions]
            self.message_lines.append(f" Possible extensions: {comma_join(available)}.")


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
        super().__init__("")
        self.message_lines = [
            f'The model type "{self.printer.error_text(model_type)}" is not supported.'
        ]
        if model_types:
            model_types = [f"{self.printer.pass_text(x)}" for x in sorted(model_types)]
            self.message_lines.append(f" Possible model types: {comma_join(model_types)}.")


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
        super().__init__("")
        self.message_lines = [
            f"Could not create a root MFA temporary directory (tried {self.printer.error_text(temporary_directory)}. ",
            f"Please specify a write-able directory via the {self.printer.emphasized_text(variable)} environment variable.",
        ]


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


class PyniniAlignmentError(G2PError):
    """
    Exception class for errors in alignment for Pynini training
    """

    def __init__(self, error_dict: Dict[str, Exception]):
        super().__init__("The following Pynini alignment jobs encountered errors:")
        self.message_lines.extend(["", ""])
        for k, v in error_dict.items():
            self.message_lines.append(self.printer.indent_string + self.printer.error_text(k))
            self.message_lines.append(
                self.printer.indent_string + self.printer.emphasized_text(str(v))
            )


class PyniniGenerationError(G2PError):
    """
    Exception class for errors generating pronunciations with Pynini
    """

    def __init__(self, error_dict: Dict[str, Exception]):
        super().__init__("The following words had errors in running G2P:")
        self.message_lines.extend(["", ""])
        for k, v in error_dict.items():
            self.message_lines.append(self.printer.indent_string + self.printer.error_text(k))
            self.message_lines.append(
                self.printer.indent_string + self.printer.emphasized_text(str(v))
            )


class PhonetisaurusSymbolError(G2PError):
    """
    Exception class for errors generating pronunciations with Pynini
    """

    def __init__(self, symbol, variable):
        super().__init__("")
        self.message_lines = [
            f'The symbol "{symbol}" is reserved for "{variable}", but is found in the graphemes or phonemes of your dictionary.',
            f'Please re-run and specify another symbol that is not used in your dictionary with the "--{variable}" flag.',
        ]


class LMError(MFAError):
    """
    Exception class for errors in language models
    """

    pass


class LanguageModelNotFoundError(LMError):
    """
    Exception class for a language model not being found

    Parameters
    ----------
    path: str
        Path to missing language model
    """

    def __init__(self, path: str):
        super().__init__(f"Could not find a suitable language model: {path}")


class MultiprocessingError(MFAError):
    """
    Exception class for exceptions in multiprocessing workers

    Parameters
    ----------
    job_name: int
        Job identifier
    error_text:str
        Traceback for exception in worker
    """

    def __init__(self, job_name: int, error_text: str):
        super().__init__(f"Job {job_name} encountered an error:")
        self.message_lines = [f"Job {self.printer.error_text(job_name)} encountered an error:"]
        self.job_name = job_name
        self.message_lines.extend(
            [self.highlight_line(x) for x in error_text.splitlines(keepends=False)]
        )

    def highlight_line(self, line: str) -> str:
        """
        Highlight a line in traceback

        Parameters
        ----------
        line: str
            Line to highlight

        Returns
        -------
        str
            Highlighted line
        """
        emph_replacement = self.printer.emphasized_text(r"\1")
        err_replacement = self.printer.error_text(r"\1")
        line = re.sub(r"File \"(.*)\"", f'File "{emph_replacement}"', line)
        line = re.sub(r"line (\d+)", f"line {err_replacement}", line)
        return line


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

    def __init__(self, error_logs: List[str], log_file: Optional[str] = None):
        super().__init__(
            f"There were {len(error_logs)} job(s) with errors when running Kaldi binaries."
        )
        self.job_name = None
        self.error_logs = error_logs
        self.log_file = log_file
        self.refresh_message()

    def refresh_message(self) -> None:
        """Regenerate the exceptions message"""
        from montreal_forced_aligner.config import GLOBAL_CONFIG

        self.message_lines = [
            f"There were {len(self.error_logs)} job(s) with errors when running Kaldi binaries.",
            "See the log files below for more information.",
        ]
        for error_log in self.error_logs:
            self.message_lines.append(error_log)
            if GLOBAL_CONFIG.current_profile.verbose:
                with open(error_log, "r", encoding="utf8") as f:
                    for line in f:
                        self.message_lines.append(line.strip())
        if self.log_file:
            self.message_lines.append(
                f" For more details, please check {self.printer.error_text(self.log_file)}"
            )

    def append_error_log(self, error_log: str) -> None:
        """
        Add error log for the exception

        Parameters
        ----------
        error_log: str
            Path to error log
        """
        self.error_logs.append(error_log)
        self.refresh_message()

    def update_log_file(self) -> None:
        """
        Update the log file output

        Parameters
        ----------
        logger: logging.Logger
            Logger
        """

        logger = logging.getLogger("mfa")
        if logger.handlers:
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    self.log_file = handler.baseFilename
                    break
        self.refresh_message()
