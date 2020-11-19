
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

Steps to run the validation utility:

1. Provided the steps in :ref:`installation` have been completed and you are in the same Conda/virtual environment that
   MFA was installed in.

2. Run the following command, substituting the arguments with your own paths:

  .. code-block:: bash

     mfa validate corpus_directory dictionary_path [optional_acoustic_model_path]

The ``corpus_directory`` argument should be a full path to the corpus to validate, following the proper :ref:`data_format`.
The ``dictionary_path`` should be a full path to the pronunciation dictionary you want to use with
the corpus, following the proper :ref:`dictionary`.  The optional ``acoustic_model_path`` can be used
to test alignment as well as flag potential transcription issues if ``--test_transcriptions`` is present.
The ``acoustic_model_path`` should be either a full path to an acoustic model you've trained, or you can use one of the
:ref:`pretrained_acoustic`.

Extra options to the validation utility:

.. option:: -s NUMBER
               --speaker_characters NUMBER

   Number of characters to use to identify speakers; if not specified,
   the aligner assumes that the directory name is the identifier for the
   speaker.  Additionally, it accepts the value ``prosodylab`` to use the second field of a ``_`` delimited file name,
   following the convention of labelling production data in the ProsodyLab at McGill.

.. option:: -t DIRECTORY
               --temp_directory DIRECTORY

   Temporary directory root to use for aligning, default is ``~/Documents/MFA``

.. option:: -j NUMBER
               --num_jobs NUMBER

  Number of jobs to use; defaults to 3, set higher if you have more
  processors available and would like to align faster

.. option:: --ignore_acoustics

   Prevent validation of feature generation and initial alignment.  Using this flag will make validation much faster.

.. option:: --test_transcriptions

   If flagged, the validation utility will construct simple unigram language model and attempt to decode each segment to
   be aligned.  Segments are flagged if the decoded transcriptions contain deviations from the original transcriptions.
   This is largely experimental feature that may be useful, but may not be always reliable.  Cannot be flagged at the
   same time as ``--ignore_acoustics``