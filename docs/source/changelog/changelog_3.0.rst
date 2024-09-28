
.. _changelog_3.0:

*************
3.0 Changelog
*************


3.0.8
-----

- Fixed a compatibility issue with models trained under version 1.0 and earlier

3.0.7
-----

- Add check for current version vs latest version on run
- Added :code:`--final_clean` flag to clean temporary files at the end of each run, along with a :code:`--always_final_clean` flag for :code:`mfa configure`
- Removed dependencies on :code:`sox` and :code:`ffmpeg` as audio loading is done through :code:`librosa` in :code:`kalpy`
- Removed poorly aligned files in subset from further training
- Fixed an issue where specified words for cutoff modeling

3.0.6
-----

- Fixed an issue where alignment analysis would not produce data for speech log likelihood and phone duration deviation
- Changed phone duration deviation metric to be maximum duration deviation rather than average across all phones in the utterance
- Fixed a crash when an empty phone set was specified in phone groups configuration files
- Fixed a crash when when using the :code:`--language` flag with values other than :code:`japanese`, :code:`thai`, :code:`chinese` or :code:`korean`

3.0.5
=====

- Added :code:`mfa_update` command to better sync changes across dependencies
- Updated how calculated properties are loaded to fix crashes in Anchor
- Change when alignments are analyzed in training

3.0.4
=====

- Fixed issue with github token set in the environment not being respected
- Changed ordering of g2p output from corpora to be based on word frequency rather than alphabetical
- Changed duration deviation to save the max z-scored duration, rather than be the average over all phones
- Update default punctuation markers to cover Arabic script punctuation

3.0.3
=====

- Fixed regression for not merging clitic words when textgrid cleanup is disabled
- Fixed issue with copying files when symlinks are not possible on windows
- Fixed an issue with using G2P models during training/alignment
- Changed default feature config to set :code:`use_energy=True` and :code:`dithering=0.0001`
- Updated tokenization when lower casing to remove extra dot for capital :code:`i` in Turkish
- Fix an issue where special disambiguation symbols were not always in the phone table

3.0.2
=====

- Added support for :code:`--phone_groups_path` and :code:`--rules_path` to :ref:`validating_data`
- Added support for speechbrain 1.0 release
- Allow alignment with older models that don't have a dedicated speaker-independent :code:`.alimdl` model
- Fixed a bug in loading lexicon compilers
- Updated default feature configuration to remove dithering and use energy_floor=1.0, following `torchaudio's implementation <https://github.com/pytorch/audio/issues/371>`_

3.0.1
=====

- Fixed an issue where pool size would be too low for number of jobs
- Fixed an issue with specifying :code:`--phone_groups_path` causing a crash

3.0.0
=====

- Fixed a regression where :code:`--dither` was not being passed correctly
- Fixed a bug on Windows when symlink permissions were not present

3.0.0rc2
========

- Add support for per-dictionary g2p models during acoustic model training and alignment
- Change Chinese language support to require :xref:`dragonmapper`
- Fixed bug in TextGrid generation for incorrect number of intervals

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
- Add tokenization support for Chinese languages when :xref:`spacy-pkuseg` and :xref:`hanziconv` are installed via :code:`pip install spacy-pkuseg hanziconv dragonmapper`
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
