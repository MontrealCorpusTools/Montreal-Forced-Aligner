from __future__ import annotations
from colorama import Fore, Back, Style
from .helper import comma_join

class MFAError(Exception):
    """
    Base exception class
    """
    def __init__(self, *args, **kwargs):
        from .config import USE_COLORS
        self.red_text = ''
        self.bright_text = ''
        self.green_text = ''
        self.reset_text = ''
        if USE_COLORS:
            self.red_text = Fore.RED
            self.bright_text = Style.BRIGHT
            self.green_text = Fore.GREEN
            self.reset_text = Style.RESET_ALL
        self.message = ''
        if args and isinstance(args[0], str):
            self.message = args[0]

    def error_text(self, text: str) -> str:
        return f'{self.red_text}{text}{self.reset_text}'

    def emphasized_text(self, text: str) -> str:
        return f'{self.bright_text}{text}{self.reset_text}'

    def pass_text(self, text: str) -> str:
        return f'{self.green_text}{text}{self.reset_text}'

    def __str__(self) -> str:
        return f"{self.error_text(type(self).__name__)}: {self.message}"

class ThirdpartyError(MFAError):
    def __init__(self, binary_name, open_fst=False, open_blas=False, libc=False, sox=False):
        super().__init__()
        self.message = f"Could not find '{self.error_text(binary_name)}'. "
        extra = "Please ensure that you have downloaded the correct binaries."
        if open_fst:
            extra = (f"Please ensure that you are in an environment that has the {self.emphasized_text('openfst')} conda package installed, "
                     f"or that the {self.emphasized_text('openfst')} binaries are on your path if you compiled them yourself.")
        elif open_blas:
            extra = (f"Try installing {self.emphasized_text('openblas')} via system package manager or verify it's on your system path?")
        elif libc:
            extra = (f"You likely have a different version of {self.emphasized_text('glibc')} than the precompiled binaries use. "
                     f"Try compiling {self.emphasized_text('Kaldi')} on your machine and collecting the binaries via the "
                     f"`{self.green_text('mfa thirdparty kaldi')}` command.")
        elif sox:
            self.message = ''
            extra = (f"Your version of {self.emphasized_text('sox')} does not support the file format in your corpus. "
                    f"Try installing another version of {self.emphasized_text('sox')} with support for {self.error_text(binary_name)}.")
        self.message += extra

# Model Errors

class ModelError(MFAError):
    pass

class ModelLoadError(ModelError):
    def __init__(self, path: str):
        super(ModelLoadError, self).__init__()
        self.message = f"The archive {self.error_text(path)} could not be parsed as an MFA model"

# Dictionary Errors

class DictionaryError(MFAError):
    """
    Class for errors in creating Dictionary objects
    """
    pass

class NoDefaultSpeakerDictionaryError(DictionaryError):
    """
    Class for errors in creating Dictionary objects
    """

    def __init__(self):
        super().__init__()
        self.message = f'No "{self.error_text("default")}" dictionary was found.'


class DictionaryPathError(DictionaryError):
    """
    Class for errors in locating paths for Dictionary objects
    """

    def __init__(self, input_path: str):
        super().__init__()
        self.message = f'The specified path for the dictionary ({self.error_text(input_path)}) was not found.'


class DictionaryFileError(DictionaryError):
    """
    Class for errors in locating paths for Dictionary objects
    """

    def __init__(self, input_path: str):
        super().__init__()
        self.message = f'The specified path for the dictionary ({self.error_text(input_path)}) is not a file.'


# Corpus Errors

class CorpusError(MFAError):
    """
    Class for errors in creating Corpus objects
    """
    pass


class SampleRateMismatchError(CorpusError):
    """
    Class for errors in different sample rates
    """
    pass


class CorpusReadError(CorpusError):
    """
    Class for errors in different sample rates
    """
    def __init__(self, file_name: str):
        self.file_name = file_name



class TextParseError(CorpusReadError):
    """
    Class for errors parsing lab and txt files
    """
    def __init__(self, file_name):
        self.file_name = file_name


class TextGridParseError(CorpusReadError):
    """
    Class capturing TextGrid reading errors
    """
    def __init__(self, file_name, error):
        self.file_name = file_name
        self.error = error


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
    """
    def __init__(self, error_logs):
        super().__init__()
        output = '\n'.join(error_logs)
        self.message = f'There were {len(error_logs)} job(s) with errors.  ' \
                  f'For more information, please see the following logs:\n\n{output}'


class AlignmentExportError(AlignmentError):
    """
    Class for errors during alignment
    """
    def __init__(self, error_dict):
        MFAError.__init__(self)

        message = 'Error was encountered in processing CTMs:\n\n'
        for key, error in error_dict.items():
            message += f'{key}:\n{error}'
        self.message = message


class NoSuccessfulAlignments(AlignerError):
    """
    Class for errors where nothing could be aligned
    """
    pass


class PronunciationAcousticMismatchError(AlignerError):
    def __init__(self, missing_phones):
        super().__init__()
        missing_phones = [f'{self.error_text(x)}' for x in sorted(missing_phones)]
        self.message = f'There were phones in the dictionary that do not have acoustic models: ' \
                  f'{comma_join(missing_phones)}'


class PronunciationOrthographyMismatchError(AlignerError):
    def __init__(self, g2p_model, dictionary):
        super().__init__()
        missing_graphs = dictionary.graphemes - set(g2p_model.meta['graphemes'])
        missing_graphs = [f'{self.error_text(x)}' for x in sorted(missing_graphs)]
        self.message = f'There were graphemes in the corpus that are not covered by the G2P model: ' \
                  f'{comma_join(missing_graphs)}'


# Command line exceptions

class ArgumentError(MFAError):
    pass


class FileArgumentNotFoundError(ArgumentError):
    def __init__(self, path):
        super().__init__()
        self.message = f'Could not find "{self.error_text(path)}".'


class PretrainedModelNotFoundError(ArgumentError):
    def __init__(self, name, model_type=None, available=None):
        super().__init__()
        extra = ''
        if model_type:
            extra += f' for {model_type}'
        message = f'Could not find a model named "{self.error_text(name)}"{extra}.'
        if available:
            available = [f'{self.pass_text(x)}' for x in available]
            message += f' Available: {comma_join(available)}.'
        self.message = message


class MultipleModelTypesFoundError(ArgumentError):
    def __init__(self, name, possible_model_types):
        super().__init__()

        possible_model_types = [f'{self.error_text(x)}' for x in possible_model_types]
        self.message = (f'Found multiple model types for "{self.error_text(name)}": {", ".join(possible_model_types)}. '
                       f'Please specify a model type to inspect.')

class ModelExtensionError(ArgumentError):
    def __init__(self, name, model_type, extensions):
        super().__init__()
        extra = ''
        if model_type:
            extra += f' for {model_type}'
        message = f'The path "{self.error_text(name)}" does not have the correct extensions{extra}.'
        if extensions:
            available = [f'{self.pass_text(x)}' for x in extensions]
            message += f' Possible extensions: {comma_join(available)}.'
        self.message = message


class ModelTypeNotSupportedError(ArgumentError):
    def __init__(self, model_type, model_types):
        super().__init__()
        message = f'The model type "{self.error_text(model_type)}" is not supported.'
        if model_types:
            model_types = [f'{self.pass_text(x)}' for x in sorted(model_types)]
            message += f' Possible model types: {comma_join(model_types)}.'
        self.message = message


class ConfigError(MFAError):
    pass


class TrainerError(MFAError):
    pass


class G2PError(MFAError):
    pass


class LMError(MFAError):
    pass

class LanguageModelNotFoundError(LMError):
    def __init__(self):
        super(LanguageModelNotFoundError, self).__init__()
        self.message = 'Could not find a suitable language model'


class KaldiProcessingError(MFAError):
    def __init__(self, error_logs, log_file=None):
        super().__init__()
        self.message = 'There was one or more errors when running Kaldi binaries.'
        if log_file is not None:
            self.message += f' For more details, please check {self.error_text(log_file)}'
        self.error_logs = error_logs
        self.log_file = log_file

    def update_log_file(self, log_file):
        self.log_file = log_file
        self.args = (f'There was one or more errors when running Kaldi binaries.'
                     f' For more details, please check {log_file}',)