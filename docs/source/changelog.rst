.. _`PR #194`: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/pull/194
.. _`PR #235`: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/pull/235
.. _`PR #288`: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/pull/288

.. _changelog:

Changelog
=========

2.0.0a22
--------

- Add support for aligning mp3 files

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

- Changed how punctuation is stripped from beginning/end of words (`PR #288`_)
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
- Fixed bug in checking for brackets in G2P (`PR #194`_)
- Updated Annotator utility (:ref:`annotator` for more details) to be generally more usable for TextGrid use cases and
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
- Optimized corpus parsing algorithm to be O(n log n) instead of O(n^2) (`PR #194`_)


1.1.0
-----

Major changes to system, see :ref:`whats_new_1_1`.

1.0.0
-----

- Added Grapheme-to-Phoneme capabilities
- Acoustic models no longer contain the dictionary they were trained with
- Dictionaries must be specified when aligning using pretrained models
- The aligner now automatically cleans the temporary directory when the previous run failed
- Added validation for types of command line arguments
- Catch and list files that could not be read using UTF-8
- Update Kaldi version to 5.1 and OpenFST version to 1.6.2 on Mac and Linux
- Add support for specifying custom non-speech annotations in pronunciation dictionary with sil and spn
- Made command line flags more consistent in spelling
- Made pretrained models for many languages available

0.8.0
-----

- Fixed an issue where aligning using pretrained models was improperly updating the original model with sparser data
- Added a flag to turn off speaker adaptation when aligning using a pretrained model
- Optimized training graph generation when aligning using a pretrained model

0.7.3
-----

- Added warning messages and log output when wav files are ignored because they have too low of a sampling rate or
  no .lab or .TextGrid file associated with them

0.7.2
-----

- Fixed an issue where speaker character flags were being ignored when parsing TextGrid files

0.7.1
-----

- Fixed an issue where the number of gaussians was set too low for triphone training

0.7.0
-----

- Fixed an issue with unicode characters not being correctly parsed when using ``--nodict``
- Fixed an issue where short intervals in TextGrid were not being properly ignored
- Added a command line option ``--temp_directory`` to allow for user specification of the
  temporary directory that MFA stores all files during alignment, with the
  default of ``~/Documents/MFA``
- Added logging directory and some logging for when utterances are ignored

0.6.3
-----

- Improved memory and time efficiency of extracting channels from stereo
  files, particularly for long sound files

0.6.2
-----

- Fixed an issue where pretrained models were not being bundled with the source code

0.6.1
-----

- Fixed an issue with Linux binaries not finding Kaldi binaries
- English models now use all of LibriSpeech dataset and not just clean
  subset (increased number of accents being the primary difference between the two)

0.6.0
-----

- Added commandline argument ``--clean`` to remove temporary files
- Added support for multiple sampling rates in a single dataset
- Fix some bugs relating to using a single process
- Fixed a bug where spaces were being inserted into transcriptions when using ``--nodict``
- Fixed a bug where having no out-of-vocabulary items would cause a crash at the end of aligning
- Fixed a bug where the frozen executable could not find the included pretrained models
- Fixed an issue where dictionaries in model outputs were binary files rather than editable text files
- Added docstrings to main classes
- Updated built in model ``english`` for the full 1000-hour LibriSpeech corpus

0.5.0
-----

- Initial release
- Prosodylab-aligner format supported
- TextGrid format supported
- Align using pretrained models supported
- Train models and align concurrently supported

