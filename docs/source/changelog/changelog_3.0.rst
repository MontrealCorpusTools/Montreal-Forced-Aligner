
.. _changelog_3.0:

*************
3.0 Changelog
*************

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
