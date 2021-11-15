
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
- Any transcription files missing .wav files
- Any .wav files missing transcription files
- Any issues reading transcription files
- Any unsupported sampling rates of .wav files
- Any unaligned files from a basic monophone acoustic model trained on the dataset (or using a supplied acoustic model),
  skipped if ``--ignore_acoustics`` is flagged
- Any files that have deviations from their original transcription to decoded transcriptions using a simple language model


.. _running_the_validator:

Running the validation utility
==============================


Command reference
-----------------

.. autoprogram:: montreal_forced_aligner.command_line.mfa:parser
   :prog: mfa
   :start_command: validate
