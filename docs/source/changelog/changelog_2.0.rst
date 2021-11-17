
.. _changelog_2.0:

*************
2.0 Changelog
*************

.. _2.0b:

Beta releases
=============

2.0.0b5
-------

- Documentation refresh! Docs now use the :xref:`pydata_sphinx_theme` and should have a better landing page and flow, as well as up to date API reference
- Some refactoring to use type hinting and abstract class interfaces (still a work in progress)


2.0.0b4
-------

- Massive refactor to a proper class-based API for interacting with MFA corpora

  - Sorry, I really do hope this is the last big refactor of 2.0
  - :class:`~montreal_forced_aligner.corpus.Speaker`, :class:`~montreal_forced_aligner.corpus.File`, and :class:`~montreal_forced_aligner.corpus.Utterance` have dedicated classes rather than having their information split across dictionaries mimicking Kaldi files, so they should be more useful for interacting with outside of MFA
  - Added :class:`~montreal_forced_aligner.multiprocessing.Job` class as well to make it easier to generate and keep track of information about different processes
- Updated installation style to be more dependent on conda-forge packages

  - Kaldi and MFA are now on conda-forge! |:tada:|

- Added a :code:`mfa model` command for inspecting, listing, downloading, and saving pretrained models, see :ref:`pretrained_models` for more information.
- Fixed a bug where saving command history with errors would throw an error of its own
- Fixed an issue where one Job could process another Job's data, result in an error
- Updated API documentation to reflect refactor changes


2.0.0b3
-------

- Fixed a bug involving non-escaped orthographic characters
- Improved SAT alignment with speaker-independent alignment model
- Fixed a bug where models would not function properly if they were renamed
- Added a history subcommand to list previous commands

2.0.0b1
-------

- Fixed bug in training (:mfa_pr:`337`)
- Fixed bug when using Ctrl-C in loading

2.0.0b0
-------

Beta release!

- Fixed an issue in transcription when using a .ARPA language model rather than one built in MFA
- Fixed an issue in parsing filenames containing spaces
- Added a ``mfa configure`` command to set global options.  Users can now specify a new default for arguments like ``--num_jobs``, ``--clean`` or ``--temp_directory``, see :ref:`configuration` for more details.
- Added a new flag for overwriting output files. By default now, MFA will not output files if the path already exists, and will instead write to a directory in the temporary directory.  You can revert this change by running ``mfa configure --always_overwrite``
- Added a ``--disable_textgrid_cleanup`` flag to disable for post-processing that MFA has implemented recently (not outputting silence labels and recombining subwords that got split up as part of dictionary look up). You can set this to be the default by running ``mfa configure --disable_textgrid_cleanup``
- Refactored and optimized the TextGrid export process to use multiple processes by default, you should be significant speed ups.
- Removed shorthand flags for ``-c`` and ``-d`` since they could represent multiple different flags/arguments.

.. _2.0a:

2.0 alpha releases
==================

2.0.0a24
--------

- Fixed some miscellaneous bugs and cleaned up old and unused code

2.0.0a23
--------

- Fix bugs in transcription and aligning with using multiple dictionaries
- Fixed an issue where filenames were output with ``-`` rather than ``_`` if they originally had them
- Changed how output text different from input text when there was a compound marker (i.e., ``-``), these should now
  have a single interval for the whole compound rather than two intervals for each subword
- Changed how OOV items are output, so they will be present in the output rather than ``<unk>``

2.0.0a22
--------

- Add support for aligning mp3 files
- Fix for log error in 0 probability entries in probabilistic lexicons
- Add support for multilingual IPA mode (see :ref:`multilingual_ipa` for more details)
- Add support for specifying per-speaker pronunciation dictionaries (see :ref:`speaker_dictionaries` for more details)
- Fixed cases where TextGrid parsing errors were misattributed to sound file issues, and these should be properly detected
  by the validator now
- Add check for system version of libc to provide a more informative error message with next steps for compiling Kaldi on
  the user's machine
- Update annotator utility to have autosave on exit
- Fixed cases where not all phones in a dictionary were present in phone_mapping
- Changed TextGrid export to not put "sp" or "sil" in the phone tier

2.0.0a21
--------

- Fixed a memory leak in corpus parsing introduced by 2.0.0a20

2.0.0a20
--------

- Fixed an issue with :code:`create_segments` where it would assue singular speakers
- Fixed a race condition in multiprocessing where the queue could finish with the jobs still running and unable to join
- Updated transcription to use a small language model for first pass decoding followed by LM rescoring in line with Kaldi recipes
- Added an optional :code:`--audio_directory` argument for finding sound files in a directory separate from the transcriptions
- Added perplexity calculations for language model training
- Updated annotator GUI to support new improvements, mainly playback of :code:`.flac` files
- Added annotator GUI functionality for showing all speaker tiers
- Added annotator GUI functionality for changing speakers of utterances by clicking and dragging them
- Updated annotator GUI to no longer aggressively zoom when selecting, merging, or splitting utterances, instead zoom
  functionality is achieved through double clicks


2.0.0a19
--------

- Fixed a bug where command line arguments were not being correctly passed to ``train`` and other commands

2.0.0a18
--------

- Changes G2P model training validation to not do a full round of training after the validation model is trained
- Adds the ability to change in alignment config yamls the punctuation, clitic, and compound marker sets used in
  sanitizing words in dictionary and corpus uses
- Changed configuration in G2P to fit with the model used in alignment, allow for configuration yamls to be passed, as
  well as arguments from command line
- Fix a bug where floating point wav files could not be parsed

2.0.0a17
--------

- Optimizes G2P model training for 0.3.6 and exposes :code:`--batch_size`, :code:`--max_iterations`, and :code:`--learning_rate`
  from the command line
- Changes where models are stored to make them specific to the alignment run rather than storing them globally in the temporary
  directory

2.0.0a16
--------

- Changed how punctuation is stripped from beginning/end of words (:mfa_pr:`288`)
- Added more logging for alignment (validating acoustic models and generating overall log-likelihood of the alignment)
- Changed subsetting features prior to initializing monophone trainer to prevent erroneous error detection
- Fixed parsing of boolean arguments on command line to be passed to aligners

2.0.0a15
--------

- Fixed a bug with dictionary parsing that misparsed clitics as <unk> words
- Added a :code:`--clean` flag for :code:`mfa g2p` and :code:`mfa train_g2p` to remove temporary files from
  previous runs
- Added support for using :code:`sox` in feature generation, allowing for use of audio files other than WAV
- Switched library for TextGrid parsing from :code:`textgrid` to :code:`praatio`, allowing support for TextGrid files in
  the short format.

2.0.0a14
--------

- Fixed a bug in running fMMLR for speaker adaptation where utterances were not properly sorted (MFA now uses dashes to
  separate elements in utterance names rather than underscores)

2.0.0a13
--------

- Updated how sample rates are handled. MFA now generates features between 80 Hz and 7800 Hz and allows downsampling and
  upsampling, so there will be no more errors or warnings about unsupported sample rates or speakers with different sample
  rates
- Fixed a bug where some options for generating MFCCs weren't properly getting picked up (e.g., snip-edges)
- (EXPERIMENTAL) Added better support for varying frame shift. In :code:`mfa align`, you can now add a flag of :code:`--frame_shift 1` to align
  with millisecond shifts between frames.  Please note this is more on the experimental side, as it increases computational
  time significantly and I don't know fully the correct options to use for :code:`self_loop_scale`, :code:`transition_scale`,
  and :code:`acoustic_scale` to generate good alignments.
- Fixed a bug in G2P training with relative paths for output model
- Cleaned up validator output

2.0.0a11
--------

- Fixed a bug in analyzing unaligned utterances introduced by changes in segment representation

2.0.0a9
-------

- Fixed a bug when loading :code:`utterance_lengths.scp` from previous failed runs
- Added the ability to generate multiple pronunciations per word when running G2P, see the extra options in
  :ref:`g2p_dictionary_generating` for more details.

2.0.0a8
-------

- Fixed a bug in generating alignments for TextGrid corpora

2.0.0a7
-------

- Upgraded dependency of Pynini version to 2.1.4, please update package versions via :code:`conda upgrade -c conda-forge openfst pynini ngram baumwelch`
  if you had previously installed MFA.
- Allowed for splitting clitics on multiple apostrophes
- Fixed bug in checking for brackets in G2P (:mfa_pr:`235`)
- Updated Annotator utility (:ref:`anchor` for more details) to be generally more usable for TextGrid use cases and
  adjusting segments and their transcriptions
- Improved handling of stereo files with TextGrids so that MFA doesn't need to generate temporary files for each channel

2.0.0a5
-------

- Fixed a bug in feature where sorting was not correct due to lack of speakers at the beginnings
  of utterances
- Fixed a bug where alignment was not performing speaker adaptation correctly
- Added a flag to :code:`align` command to disable speaker adaptation if desired
- Fixed a bug where the aligner was not properly ignored short utterances (< 0.13 seconds)
- Changed the internal handling of stereo files to use :code:`_channelX` rather than :code:`_A` and :code:`_B`
- Add a :code:`version` subcommand to output the version

2.0.0a4
-------

- Fixed a corpus parsing bug introduced by new optimized parsing system in 2.0.0a3

2.0.0a3
-------

- Further optimized corpus parsing algorithm to use multiprocessing and to load from saved files in temporary directories
- Revamped and fixed training using subsets of the corpora
- Fixed issue with training LDA systems
- Fixed a long-standing issue with words being marked as OOV due to improperly parsing clitics
- Updated logging to better capture when errors occur due to Kaldi binaries to better locate sources of issues

2.0.0
-----

Currently under development with major changes, see :ref:`whats_new_2_0`.

- Fixed a bug in dictionary parsing that caused initial numbers in pronunciations to be misparsed and ignored
- Updated sound file parsing to use PySoundFile rather than inbuilt wave module, which should lead to more informative error
  messages for files that do not meet Kaldi's input requirements
- Removed multiprocessing from speaker adaptation, as the executables use multiple threads leading to a bottleneck in
  performance.  This change should result in faster speaker adaptation.
- Optimized corpus parsing algorithm to be O(n log n) instead of O(n^2) (:mfa_pr:`194`)
