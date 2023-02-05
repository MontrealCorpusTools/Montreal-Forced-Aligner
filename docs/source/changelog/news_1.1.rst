
.. _whats_new_1_1:

What's new in 1.1
=================

Version 1.1 of the Montreal Forced Aligner represents several overhauls to the workflow and ability to customize model training and alignment.

.. attention::

   With the development of 2.0, the below sections are out of date.

.. _1_1_training_configurations:

Training configurations
-----------------------

A major new feature is the ability to specify and customize configuration for training and alignment. Prior to 1.1, the training procedure for new models was:

- Monophone training
- Triphone training
- Speaker-adapted triphone training (could be disabled)

The parameters for each of these training blocks were fixed and not changeable.

In 1.1, the following training procedures are available:

- Monophone training
- Triphone training
- LDA+MLLT training
- Speaker-adapted triphone training
- Ivector extractor training

Each of these blocks (as well as their inclusion) can be customized through a YAML config file.  In addition to training parameters,
global alignment and feature configuration parameters are available. See :ref:`configuration` for more details.


.. _1_1_data_validation:

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

.. _1_1_dictionary_generation:

Updated dictionary generation
-----------------------------

The functionality of :code:`mfa_generate_dictionary` has been expanded.

- Rather than having a :code:`--no_dict` option for alignment executables, the orthographic transcription functionality is now
  used when a G2P model is not provided to :code:`mfa_generate_dictionary`
- When a corpus directory is specified as the input path, all words will be parsed rather than just those from transcription
  files with an associated sound file
- When a text file is specified as the input path, all words in the text file will be run through G2P, allowing for a
  simpler pipeline for generating transcriptions from out of vocabulary items
