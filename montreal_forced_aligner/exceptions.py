class MFAError(Exception):
    """
    Base exception class
    """
    pass


class ThirdpartyError(MFAError):
    pass

# Dictionary Errors

class DictionaryError(MFAError):
    """
    Class for errors in creating Dictionary objects
    """
    pass


class DictionaryPathError(DictionaryError):
    """
    Class for errors in locating paths for Dictionary objects
    """

    def __init__(self, input_path):
        self.input_path = input_path
        message = 'The specified path for the dictionary ({}) was not found.'.format(input_path)
        super(DictionaryPathError, self).__init__(message)


class DictionaryFileError(DictionaryError):
    """
    Class for errors in locating paths for Dictionary objects
    """

    def __init__(self, input_path):
        self.input_path = input_path
        message = 'The specified path for the dictionary ({}) is not a file.'.format(input_path)
        super(DictionaryFileError, self).__init__(message)


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
    def __init__(self, file_name):
        self.file_name = file_name


class SampleRateError(CorpusReadError):
    """
    Class for errors in different sample rates
    """
    pass


class TextParseError(CorpusReadError):
    """
    Class for errors in different sample rates
    """
    pass


class TextGridParseError(CorpusReadError):
    """
    Class for errors in different sample rates
    """
    def __init__(self, file_name, error):
        self.file_name = file_name
        self.error = error


class WavReadError(CorpusReadError):
    """
    Class for errors in different sample rates
    """
    pass


class SoxError(WavReadError):
    """
    Class for errors in different sample rates
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
    pass


class NoSuccessfulAlignments(AlignerError):
    """
    Class for errors where nothing could be aligned
    """
    pass


class PronunciationAcousticMismatchError(AlignerError):
    def __init__(self, missing_phones):
        message = 'There were phones in the dictionary that do not have acoustic models: {}'.format(
            ', '.join(sorted(missing_phones)))
        super(PronunciationAcousticMismatchError, self).__init__(message)


class PronunciationOrthographyMismatchError(AlignerError):
    def __init__(self, g2p_model, dictionary):
        missing_graphs = dictionary.graphemes - set(g2p_model.meta['graphemes'])
        message = 'There were graphemes in the corpus that are not covered by the G2P model: {}'.format(
            ', '.join(sorted(missing_graphs)))
        super(PronunciationOrthographyMismatchError, self).__init__(message)


# Command line exceptions

class ArgumentError(MFAError):
    pass


class ConfigError(MFAError):
    pass


class TrainerError(MFAError):
    pass


class G2PError(MFAError):
    pass


class LMError(MFAError):
    pass


class KaldiProcessingError(MFAError):
    def __init__(self, error_logs, log_file=None):
        message = 'There was one or more errors when running Kaldi binaries.'
        if log_file is not None:
            message += ' For more details, please check {}'.format(log_file)
        super(KaldiProcessingError, self).__init__()
        self.error_logs = error_logs
        self.log_file = log_file

    def update_log_file(self, log_file):
        self.log_file = log_file
        self.args = ('There was one or more errors when running Kaldi binaries. For more details, please check {}'.format(log_file),)