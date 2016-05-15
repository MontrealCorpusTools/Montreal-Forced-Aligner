

class MFAError(Exception):
    pass

## Dictionary Errors

class DictionaryError(MFAError):
    pass

## Corpus Errors

class CorpusError(MFAError):
    pass

class SampleRateError(CorpusError):
    pass

## Aligner Errors

class AlignerError(MFAError):
    pass

class NoSuccessfulAlignments(AlignerError):
    pass
