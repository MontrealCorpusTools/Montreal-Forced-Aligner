
.. _news:

****
News
****

.. _whats_new_2_0:

What's new in 2.0
=================

Version 2.0 of the Montreal Forced Aligner represents several overhauls to installation and management
of commands.  See :ref:`changelog_2.0` for a more specific changes.

.. _2_0_installation_update:

Installation style
------------------

Up until now, MFA has used a frozen executable model for releases, which involves packaging MFA code along with a Python
interpreter, some system libraries, and compiled third party executables from Kaldi, OpenFST, OpenNgram, and Phonetisaurus.
The main issues with this style of distribution revolve around inefficiencies in the build system and a lack of ability to
customize the runtime for different environments and versions.

Moving forward, MFA will:

- Use standard Python packaging and be available for import in Python
- Rely on :xref:`conda_forge` for handling dependencies
- Switch to using Pynini instead of Phonetisaurus for G2P purposes, which should ease distribution and installation
- Have a :ref:`2_0_unified_cli` with subcommands for each command line function that will be available upon installation, as well as exposing the full MFA api for use in other Python scripts
- Allow for faster bug fixes that do not require repackaging and releasing frozen binaries across all platforms

.. _2_0_unified_cli:

Unified command line interface
------------------------------

Previously, MFA has used multiple separate frozen CLI programs to perform specific actions. However, as
more functionality has been added with G2P models, validation, managing pretrained models, and training
different types of models, it has become unwieldy to have separate commands for each. As such, going
forward:

- There will be a single :code:`mfa` command line utility that will be available once it is installed via pip/conda.
- Running :code:`mfa -h` will list the subcommands that can be run, along with their descriptions, see :ref:`commands` for details.

.. _2_0_anchor_gui:

Anchor annotator GUI
--------------------

Added a basic annotation GUI with features for:

- Listing processed utterances in the corpus with the ability to see which utterances have words not found in your pronunciation dictionary
- Allowing for audio playback of utterances and modification of utterance text
- Listing entries in an imported pronunciation dictionary
- Updating/adding dictionary entries
- Updating transcriptions

See also :ref:`anchor` for more information on using the annotation GUI.

.. _2.0_transcription:

Transcription
-------------

MFA now supports:

- Transcribing a corpus of sound files using an acoustic model, dictionary, and language model, see :ref:`transcribing` for
  more information.
- Training language models from corpora that have text transcriptions, see :ref:`training_lm` for more information
- Training pronunciation probability dictionaries from alignments, for use in alignment or transcription, see :ref:`training_dictionary` for more information

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


.. toctree::
   :maxdepth: 1
   :hidden:

   changelog_2.0.rst
   changelog_1.0.rst
