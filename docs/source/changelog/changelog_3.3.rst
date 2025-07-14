
.. _changelog_3.3:

*************
3.3 Changelog
*************

3.3.3
-----

- Fixed a data type mismatch issue when using whisperx with CPU
- Fixed a bug where files were being multiply transcribed when using torch models on CPU
- Fixed a python version compatibility issue with paths being saved in configuration files
- Fixed a compatibility issue with sklearn 1.5

3.3.0
-----

- Added support for incorporating existing reference alignments into acoustic model training and adaptation
- Added ReferencePhoneIntervals and ReferenceWordIntervals tables separate from those generated from alignments
- Removed dependency on Biopython's pairwise2 module for evaluating alignments
- Added signal-to-noise ratio calculation as part of alignment evaluation
- Added filters in training and adaptation based on alignment evaluation metrics
- Modified temporary files to use dictionary names instead of dictionary ids for easier debugging with multiple dictionaries
- Compatibility with Kalpy 0.7.0
