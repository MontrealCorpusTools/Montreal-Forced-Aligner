
.. _changelog_3.3:

*************
3.3 Changelog
*************

3.3.5
-----

- Added utility command for :ref:`remap_alignments`
- Refactored command line utility set up into helper functions
- Update finetuning algorithm to use Kalpy's interpolate boundary rather than a full realignment pass
- Move fine tune functionality from :class:`~montreal_forced_aligner.alignment.AlignMixin` to :class:`~montreal_forced_aligner.alignment.PretrainedAligner`
- Fixed a bug where phone duration deviation and speech log-likelihood metrics were not properly being saved after alignment :github_issue:`909`

3.3.4
-----

- Fixed a bug with cutoff modeling for small datasets
- Added command to :ref:`g2p_find_oovs`
- Fixed a bug in ``mfa transcribe_whisper`` where ``--vad`` flag was not creating TextGrid output with the segments generated from VAD
- Added validation check for corpus directories to prevent overwriting of files in the MFA_ROOT_DIR temporary directory
- Updated phone duration mean and standard deviation calculation

3.3.3
-----

- Fixed a data type mismatch issue when using whisperx with CPU
- Fixed a bug where files were being multiply transcribed when using torch models on CPU
- Fixed a python version compatibility issue with paths being saved in configuration files
- Fixed a compatibility issue with sklearn 1.5

3.3.2
-----

- Clarify error message when modules removed in Python 3.13 are not found.

3.3.1
-----

- Added better error handling for multiprocessing functions
- Added more consistent handling of reference phone and word intervals

3.3.0
-----

- Added support for incorporating existing reference alignments into acoustic model training and adaptation
- Added ReferencePhoneIntervals and ReferenceWordIntervals tables separate from those generated from alignments
- Added utility command for :ref:`remap_dictionary`
- Removed dependency on Biopython's pairwise2 module for evaluating alignments
- Added signal-to-noise ratio calculation as part of alignment evaluation
- Added filters in training and adaptation based on alignment evaluation metrics
- Modified temporary files to use dictionary names instead of dictionary ids for easier debugging with multiple dictionaries
- Compatibility with Kalpy 0.7.0
