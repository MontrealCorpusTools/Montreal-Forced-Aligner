
.. _changelog_3.0:

*************
3.0 Changelog
*************

3.0.0rc1
========

- Fixed a bug related to fMLLR computation in kalpy that was causing a degradation in aligner performance
- Improved memory usage for large corpora when generating MFCCs
- Improved subset logic in acoustic model training to ensure all speakers in the subset have at least 5 utterances for better training
- Fixed a bug in triphone training initialization that was causing a degradation in aligner performance
- Reimplemented multiprocessing in addition to threading from 3.0.0a1
- Made logging more verbose for acoustic model training
- Improved subset logic for G2P training and validation splits to ensure low-frequency graphemes and phones are reliably in the training data
- Added better validation for phone groups files in acoustic model training
- Added better validation for phone mapping files in alignment evaluation
- Add tokenization support for Chinese languages when :xref:`spacy-pkuseg` and :xref:`hanziconv` are installed via :code:`pip install spacy-pkuseg hanziconv`
- Add tokenization support for Korean when :xref:`python-mecab-ko` and :xref:`jamo` are installed via :code:`pip install python-mecab-ko jamo`
- Add tokenization support for Thai when :xref:`pythainlp` is installed via :code:`pip install pythainlp`
- Fixed a bug where pronunciations below the OOV count threshold were being exported at the end of acoustic model training
- Fixed a feature generation error when using MFCC+pitch features
- Changed debug output for evaluation mode in G2P model training to only output incorrect entries
- Added :code:`--model_version` parameter for all model training commands to override using MFA's version
- Optimized TextGrid exporting

3.0.0a8
=======

- Fixed an issue in not normalizing utterance and speaker xvectors from speechbrain
- Bug fixes for integration with Anchor

3.0.0a7
=======

- Fixed an issue where using relative paths could delete the all MFA temporary files with :code:`--clean`
- Fixed an issue where "<eps>" in transcript to force silence was inserting phones for OOVs rather than silence

3.0.0a6
=======

- Added support for generating pronunciations during training and alignment via :code:`--g2p_model_path`
- Added support for Japanese tokenization through sudachipy
- Fixed a crash in fine tuning
- Added functionality for allowing a directory to be passed as the output path for :ref:`align_one`

3.0.0a5
=======

- Updated for :xref:`kalpy` version 0.5.5
- Updated :code:`--single_speaker` mode to not perform speaker adaptation
- Added documentation for :ref:`concept_speaker_adaptation`

3.0.0a4
=======

- Separated out segmentation functionality into :ref:`create_segments` and :ref:`create_segments_vad`
- Fixed a bug in :ref:`align_one` when specifying a :code:`config_path`

3.0.0a3
=======

- Refactored tokenization for future spacy use

3.0.0a2
=======

- Revamped how configuration is done following change to using threading instead of multiprocessing

3.0.0a1
=======

- Add dependency on :xref:`kalpy` for interacting for Kaldi
- Add command for :ref:`align_one`
- Migrate to threading instead of multiprocessing to avoid serializing Kalpy objects
