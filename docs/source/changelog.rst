.. _`PR #194`: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/pull/194

.. _changelog:

Changelog
=========

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

