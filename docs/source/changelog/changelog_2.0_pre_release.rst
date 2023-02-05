
.. _changelog_2.0_pre:

*************************
2.0 Pre-release Changelog
*************************

.. _2.0r:

Release candidates
==================

2.0.0rc9
--------

- Fixed a bug where exporting TextGrids would fail if any were empty :github_issue:`459`

2.0.0rc8
--------

- Fixed a bug where G2P output was not correctly converted to strings :github_issue:`448`
- Fixed a bug where specifying conda or temporary directories with spaces would cause crashes :github_issue:`450`
- Fixed a crash with unspecified github_token values for ``mfa model`` commands
- Added a utility function for :ref:`validating_dictionaries`
- Fixed a bug where errors in multiprocessing workers were not properly raised by the main thread, obscuring the source of errors :github_issue:`452`
- Fixed an error in parsing text files in corpora for G2P generation or language model training :github_issue:`449`
- Added an experimental training flag for training a G2P model as part of the acoustic model training
- Fixed a bug where models trained in 1.0 would not use speaker adaptation during alignment or transcription
- Add support for exporting original text alongside word and phone alignments :github_issue:`414`
- Fixed an issue with transcribing using multiple dictionaries

2.0.0rc7
--------

- Fixed a bug where silence correction was not being calculated correctly
- Fixed a bug where sample rate could not be specified when not using multiprocessing :github_pr:`444`
- Fixed an incompatibility with the Kaldi version 1016 where BLAS libraries were not operating in single-threaded mode
- Further optimized large multispeaker dictionary loading
- Fixed a bug where subsets were not properly generated when multiple dictionaries were used

2.0.0rc6
--------

- Reverted the default export type to ``long_textgrid``, which can be changed to ``short_textgrid`` or ``json`` via the ``--output_format`` flag for commands that export TextGrids :github_issue:`434`
- Added more information for when malformed dictionary lines fail to parse (i.e., lines with just tabs on them) :github_issue:`411`
- Fixed a bug where phones with underscores in them would cause export to crash :github_issue:`432`
- Changed the overwrite behavior in export to specifically avoid overwriting input files, rather than testing the existence of the overall output directory :github_issue:`431`
- Added additional initial check to ensure that Kaldi and OpenFst binaries can be successfully invoked, rather than throwing an unintuitive error during feature creation
- Optimized initial load and TextGrid export :github_issue:`437` and :github_issue:`249`
- Allow for dictionaries with the same base name in different locations to be used side-by-side :github_issue:`417`
- Fixed a bug where initial silence was not being properly handled if silence probability training had not been done
- Removed PronunciationDictionaryMixin and PronunciationDictionary classes and refactored functionality into :class:`~montreal_forced_aligner.dictionary.multispeaker.MultispeakerDictionaryMixin` and :class:`~montreal_forced_aligner.db.Dictionary`
- Fixed a bug where alignment models would not be adapted during adaptation :github_issue:`421`

2.0.0rc5
--------

- Fixed a bug where a list of downloadable models wasn't getting output for commands like ``mfa models download acoustic``
- Added option to specify ``--output_format`` for exporting alignments for ``short_textgrids`` (the default to save space), ``long_textgrids`` (original behavior), or ``json``

2.0.0rc4
--------

- Added ``--quiet`` flag to suppress printing output to the console
- Added ability to specify ``pronunciation_probabilities`` in training blocks where probabilities of pronunciation variants and their probabilities of appearing before/after silence will be calculated based on alignment at that stage.  The lexicon files will be regenerated and use these probabilities for later training blocks
- Added a flag to export per-pronunciation silence probabilities to :ref:`training_dictionary`
- Added a flag to :ref:`transcribing` for specifying the language model weight and word insertion penalties to speed up evaluation of transcripts
- Added a final SAT training block equivalent to the :kaldi_steps:`train_quick` script
- Added early stopping of SAT training blocks if the corpus size is below the specified subset (at least two rounds of SAT training will be performed)
- Refactored how transcription parsing is done, so that you can specify word break characters other than whitespace (i.e., instances of ``.`` or ``?`` in embedded in words that are typos in the corpus)
- Refactored quotations and clitic markers, so if there happens to be a word like ``kid'``, MFA can recover the word ``kid`` from it.  If there is no word entry for ``kid`` or ``kid'`` is in the dictionary, the apostrophe will be kept.
- Refactored the ``--test_transcription`` functionality of :ref:`validating_data` to use small language models built from all transcripts of a speaker, mixed with an even smaller language model per utterance, following :kaldi_steps:`cleanup/make_biased_lm_graphs`.
- Refactored how internal storage is done to use a sqlite database rather than having everything in memory.  Bigger corpora should not need as much memory when aligning/training.
- Fixed an issue in lexicon construction where explicit silences were not being respected (:github_issue:`392`)
- Fixed an issue in training where initial gaussians were not being properly used
- Changed the behavior of assigning speakers to jobs, so that it now tries to balance the number of utterances across jobs
- Changed the default topology to allow for more variable length phones (minimum duration is now one frame, 10ms by default)
- Changed how models and dictionaries are downloaded with the changes to the `MFA Models <https://mfa-models.readthedocs.io/>`_
- Added the ability to use pitch features for models, with the ``--use_pitch`` flag or configuration option
- Added a ``[bracketed]`` word that will capture any transcriptions like ``[wor-]`` or ``<hes->``, as these are typically restarts, hesitations, speech errors, etc that have separate characteristics compared to a word that happen to not be in the dictionary.  The same phone is used for both, but having a separate word symbol allows silence probabilities to be modelled separately.
- Added words for ``[laugh]`` and ``[laughter]`` to capture laughter annotations as separate from both OOV ``<unk>`` items and ``[bracketed]`` words.  As with ``[bracketed]``, the laughter words use the same ``spn`` phone, but allow for separate silence probabilities.
- Fixed a bug where models trained in earlier version were not correctly reporting their phone set (:github_issue:`422`)
- Fixed a bug where TextGrids were never exported to the specified output directory with out ``--overwrite`` (:github_issue:`408`)
- Fixed a bug where spaces in sound file names would throw an error for that file (:github_issue:`407`)


2.0.0rc3
--------
- Fixed a bug where textgrids weren't being properly generated following training
- Fixed a bug where commands were not always respecting ``--overwrite``
- Fixed a bug where not all words in multispeaker dictionaries would be parsed
- Improved transcription accuracy calculation to account for compounds and clitics
- Fixed a crash when subsetting corpora that did not all have transcriptions

2.0.0rc2
--------
- Added configuration parameter (``ignore_case=False``) to allow for disabling the default behavior of making all text and lexicon entries lower case
- Added some metadata about training data to acoustic models

2.0.0rc1
--------

- Getting closer to stable release!
- Fixed some bugs in how transcription and alignment accuracy were calculated
- Added additional information to evaluation output files
- Added file listing average per-frame log-likelihoods by utterance for alignment
- Fixed a bug where having "<s>" in a transcript would cause MFA to crash

.. _2.0b:

Beta releases
=============

2.0.0b11
--------

- Re-optimized corpus loading following the switch to a more class-based API.
- Optimized validation, particularly when acoustics are being ignored
- Added better progress bars for corpus loading, acoustic modeling, G2P training, transcription and alignment
- Changed the default behavior of G2P generation to use a threshold system rather than returning a single top pronunciation.  The threshold defaults to 0.99, but can be specified through ``--g2p_threshold``.  Specifying number of pronunciations will override this behavior (use ``--num_pronunciation 1`` for the old behavior).
- Changed the behavior of G2P evaluation to check whether the generated hypothesis is in the golden pronunciation set, so languages with pronunciation variation will be less penalized in evaluation
- Added :class:`~montreal_forced_aligner.data.WordData` and :class:`~montreal_forced_aligner.db.Pronunciation` data classes
- Refactored and simplified TextGrid export process
- Removed the ``multilingual_ipa`` mode in favor of a more general approach to better modeling phones
- Added functionality to evaluate alignments against golden alignment set
- Added the ability to compare alignments to a reference aligned, such as human annotated data. The evaluation will compute overlap score (sum of difference in aligned phone boundaries versus the reference phone boundaries) and overall phone error rate for each utterance.

2.0.0b10
--------

- Changed the functionality of validating dictionary phones and acoustic model phones so that the aligner will simply ignore pronunciations containing phones not in the acoustic model (and print a warning).  The validator utility will provide further detail on what was ignored.
- Fixed a bug where evaluation of training G2P models was not actually triggered
- Refactored PairNGramAligner into the :class:`~montreal_forced_aligner.g2p.trainer.PyniniTrainer` class to improve logging output
- Changed the starting index of training blocks with the same name. Old behavior was ``sat``, ``sat1``, ``sat2``, etc.  The new behavior is ``sat``, ``sat2``, ``sat3``, etc.
- Revert a change with how sets, roots and extra questions are handled

2.0.0b9
-------

- Fixed a bug where unknown word phones were showing up as blank
- Fixed a bug where TextGrid export would hang
- Fixed compatibility issues with Python 3.8
- Added logging for when configuration parameters are ignored
- Added some functionality from the LibriSpeech recipe for triphone training with Arpabet

  - Not sure if it'll improve anything, but I'll run some tests and maybe extend it to other phone sets

- Added better logging to TextGrid export
- Added new classes for managing collections of utterances, speakers, and files
- Fixed a bug where oovs were not being properly reported by the validation tool

2.0.0b8
-------

- Refactored internal organization to rely on mixins more than monolithic classes, and moved internal functions to be organized by what they're used for instead of the general type.

  - For instance, there used to be a ``montreal_forced_aligner.multiprocessing`` module with ``alignment.py``, ``transcription.py``, etc that all did multiprocessing for various workers.  Now that functionality is located closer to where it's used, i.e. ``montreal_forced_aligner.transcription.multiprocessing``.
  - Mixins should allow for more easy extension to new use cases and allow for better configuration

- Updated documentation to reflect the refactoring and did a pass over the User Guide
- Added the ability to change the location of root MFA directory based on the ``MFA_ROOT_DIR`` environment variable
- Fixed an issue where the version was incorrectly reported as "2.0.0"

2.0.0b5
-------

- Documentation refresh! Docs now use the :xref:`pydata_sphinx_theme` and should have a better landing page and flow, as well as up to date API reference
- Some refactoring to use type hinting and abstract class interfaces (still a work in progress)


2.0.0b4
-------

- Massive refactor to a proper class-based API for interacting with MFA corpora

  - Sorry, I really do hope this is the last big refactor of 2.0
  - montreal_forced_aligner.corpus.classes.Speaker, :class:`~montreal_forced_aligner.corpus.classes.FileData`, and :class:`~montreal_forced_aligner.corpus.classes.UtteranceData` have dedicated classes rather than having their information split across dictionaries mimicking Kaldi files, so they should be more useful for interacting with outside of MFA
  - Added :class:`~montreal_forced_aligner.corpus.multiprocessing.Job` class as well to make it easier to generate and keep track of information about different processes
- Updated installation style to be more dependent on conda-forge packages

  - Kaldi and MFA are now on conda-forge! :fas:`party-horn`

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
- Add support for multilingual IPA mode
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
