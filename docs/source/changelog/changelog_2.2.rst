
.. _changelog_2.2:

*************
2.2 Changelog
*************

2.2.4
=====

- Fixes an issue where some directories in Common Voice Japanese were causing FileNotFound errors for sound files

2.2.3
=====

- Update terminal printing to use :mod:`rich` rather than custom logic
- Prevented the tokenizer utility from processing of text files that don't have a corresponding sound file

2.2.2
=====

- Fixed a rounding issue in parsing sox output for sound file duration
- Added ``--dictionary_path`` option to :ref:`g2p_dictionary_generating` to allow for generating pronunciations for just those words that are missing in a dictionary
- Added ``add_words`` subcommand to :ref:`pretrained_models` to allow for easy adding of words and pronunciations from :ref:`g2p_dictionary_generating` to pronunciation dictionaries

2.2.1
=====

- Fixed a couple of bugs in training Phonetisaurus models
- Added training of Phonetisaurus models for tokenizer

2.2.0
=====

- Add support for training tokenizers and tokenization
- Migrate most os.path functionality to pathlib
