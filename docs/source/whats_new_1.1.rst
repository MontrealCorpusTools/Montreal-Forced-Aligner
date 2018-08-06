
.. _whats_new_1.1:

*****************
What's new in 1.1
*****************

Version 1.1 of the Montreal Forced aligner represents several overhauls to the workflow and customizability of alignment.

.. _1.1_training_configurations:

Training configurations
-----------------------



.. _1.1_data_validation:

Data validation
---------------

In version 1.0, data validation was done as part of alignment, with user input whether alignment should be stopped if
problems were detected.  In version 1.1, all data validation is done through a separate executable :code:`mfa_validate_dataset`
(see :ref:`validating_data` for more details on usage).  Validating the dataset consists of:

- Checking for out of vocabulary items between the dictionary and the corpus
- Checking for read errors in transcription files
- Checking for transcriptions without sound files and sound files without transcriptions
- Checking for issues in feature generation (can be disabled for speed)
- Checking for issues in aligning a simple monophone model (can be disabled for speed)
- Checking for transcription errors using a simple unigram language model of common words and words in the transcript
  (disabled by default)

The user should now run :code:`mfa_validate_dataset` first and fix any issues that they perceive as important.
The alignment executables will print a warning if any of these issues are present, but will perform alignment without
prompting for user input.

.. _1.1_dictionary_generation:

Updated dictionary generation
-----------------------------

The functionality of :code:`mfa_generate_dictionary` has been expanded.

- Rather than having a :code:`--no_dict` option for alignment executables, the orthographic transcription functionality is now
  used when a G2P model is not provided to :code:`mfa_generate_dictionary`
- When a corpus directory is specified as the input path, all words will be parsed rather than just those from transcription
  files with an associated sound file
- When a text file is specified as the input path, all worrds in the text file will be run through G2P, allowing for a
  simpler pipeline for generating transcriptions from out of vocabulary items