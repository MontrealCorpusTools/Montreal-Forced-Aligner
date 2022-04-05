
.. _validating_data:

***************
Validating data
***************

The validation utility will perform the basic set up that alignment would perform, but analyzes and reports any issues
that the user may want to fix.

First, the utility parses the corpus and dictionary, prints out summary information about the corpus,
and logs any of the following issues:

- If there are any words in transcriptions that are not in the dictionary, these are logged as out-of-vocabulary items (OOVs).
  A list of these OOVs and which utterances they appear in are saved to text files.
- Any issues reading sound files
- Any issues generating features, skipped if ``--ignore_acoustics`` is flagged
- Mismatches in sound files and transcriptions
- Any issues reading transcription files
- Any unaligned files from trial alignment run, skipped if ``--ignore_acoustics`` is flagged
  - If no acoustic model is specified, a monophone model is trained for testing alignment

- Any files that have deviations from their original transcription to decoded transcriptions using a simple language model when ``--test_transcriptions`` is supplied
  - Ngram language models for each speaker are generated and merged with models for each utterance for use in decoding utterances, which may help you find transcription or data inconsistency issues in the corpus

.. warning::

   As the functionality in ``--test_transcriptions`` relies on ngram language modelling, it is not available natively on Windows outside of the Windows Subsystem for Linux.


.. _running_the_validator:

Running the validation utility
==============================


Command reference
-----------------

.. autoprogram:: montreal_forced_aligner.command_line.mfa:parser
   :prog: mfa
   :start_command: validate
   :groups:
