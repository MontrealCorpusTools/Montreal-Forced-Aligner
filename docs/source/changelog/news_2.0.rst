
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
