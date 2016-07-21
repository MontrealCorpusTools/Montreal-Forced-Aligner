.. Montreal Forced Aligner documentation master file, created by
   sphinx-quickstart on Wed Jun 15 13:27:38 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Common errors (and how to fix them)
===================================

1. ``FileNotFoundError: [Errno 2] No such file or directory: '~/Documents/MFA/[corpus name]/train/split3/cmvndeltafeats.0'``:

    You have to delete the old folder for your training data in the MFA
    folder after a crash during training.  For example, if you were aligning
    ``~/2_French_training``, there would be a folder ``~/Documents/MFA/2_French_training``
    that the aligner created. Rerun the aligner with the ``--clean`` option.

2. ``KeyError`` for wav filenames:
    Error when the folder you're aligning (or its subfolders, i.e., old
    .lab files in a subfolder) has files that are missing something crucial
    such as a corresponding .lab/.wav file.  This error could also indicate
    that utterance files might not have a unique name.

3. ``OSError: [Errno 24] Too many open files``
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
