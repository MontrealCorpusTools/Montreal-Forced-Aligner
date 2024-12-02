
.. _changelog_3.2:

*************
3.2 Changelog
*************

3.2.1
-----

- Changed unicode normalization to default to composed forms unless overridden by :code:`--unicode_decomposition true`

3.2.0
-----

- Added :code:`--subset_word_count` parameter to :ref:`train_acoustic_model` to add a minimum word count for an utterance  to be included in training subsets
- Added :code:`--minimum_utterance_length` parameter to :ref:`train_acoustic_model` to add a minimum word count for an utterance to be included in training at all
- Improved memory usage in compiling training graphs for initial subsets
- Add support for transcription via whisperx and speechbrain models
- Update text normalization to normalize to decomposed forms
- Compatibility with Kalpy 0.6.7
