.. _changelog:

Changelog
=========

0.5.1
-----

Not yet released

- Added commandline argument ``--clean`` to remove temporary files
- Added support for multiple sampling rates in a single dataset
- Fix some bugs relating to using a single process
- Fixed a bug where spaces were being inserted into transcriptions when using ``--nodict``
- Fixed a bug where having no out-of-vocabulary items would cause a crash at the end of aligning
- Fixed a bug where the frozen executable could not find the included pretrained models
- Fixed an issue where dictionaries in model outputs were binary files rather than editable text files
- Added docstrings to main classes

0.5.0
-----

- Initial release
- Prosodylab-aligner format supported
- TextGrid format supported
- Align using pretrained models supported
- Train models and align concurrently supported

