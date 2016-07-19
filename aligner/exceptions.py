

class MFAError(Exception):
    '''
    Base exception class
    '''
    pass

## Dictionary Errors

class DictionaryError(MFAError):
    '''
    Class for errors in creating Dictionary objects
    '''
    pass

## Corpus Errors

class CorpusError(MFAError):
    '''
    Class for errors in creating Corpus objects
    '''
    pass

class SampleRateError(CorpusError):
    '''
    Class for errors in different sample rates
    '''
    pass

## Aligner Errors

class AlignerError(MFAError):
    '''
    Class for errors during alignment
    '''
    pass

class NoSuccessfulAlignments(AlignerError):
    '''
    Class for errors where nothing could be aligned
    '''
    pass
