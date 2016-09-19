.. _changelog:

Changelog
=========

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

