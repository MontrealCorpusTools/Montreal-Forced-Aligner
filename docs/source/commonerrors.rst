.. Montreal Forced Aligner documentation master file, created by
   sphinx-quickstart on Wed Jun 15 13:27:38 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Common errors (and how to fix them)
===================================

1. ``OSError: [Errno 24] Too many open files``
    This is an error that pops up when speakers are probably not properly
    specified.  If there are not enough speakers, and the dataset is large enough,
    Mac and Linux have issues.  Make sure the number of speakers reported
    when setting up the corpus matches the number of speakers that should
    be there.


<unk> words in the output
`````````````````````````

See the files ``oovs_found.txt`` and ``utterance_oovs.txt`` in the
output directory for a list of all out of vocabulary words and a list
of utterances with out of vocabulary words in those utterances.
