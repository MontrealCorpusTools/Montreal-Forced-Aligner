
.. _changelog_2.0:

*************
2.0 Changelog
*************

2.0.6
=====

- Added G2P and language model training support to Windows
- Fixed a bug where exporting trained models to the current working directory would not work :github_issue:`494`
- Fixed a crash in exporting transcriptions to TextGrids
- Added support for parsing out longer quoted strings :github_issue:`492`
- Fix error message for files with no file extensions :github_issue:`495`
- Fix PhoneSetType error for some models trained on earlier versions :github_issue:`496` and :github_issue:`484`

2.0.5
=====

- Standardize :ref:`dictionary_format` to require tab delimitation between orthography, pronunciations, and any probabilities in the dictionary :github_issue:`478`
- Fixed a bug in pronunciation probability estimation when silence words are explicitly transcribed :github_issue:`476`
- Fixed an optimization bug introduced when fixing sparse job/subset combos

2.0.4
=====

- Bug fix for phonetisaurus training error in 2.0.2

2.0.2
=====

- Optimized Phonetisaurus training regime for phone and grapheme orders greater than 1
- Fixed a bug in parsing dictionaries that included whitespace as part of the word
- Fixed a bug in Phonetisaurus generation where insertions and deletions were not being properly generated
- Changed the default alignment separator for Phonetisaurus to ``;`` instead of ``}`` (shouldn't conflict with most phone sets) and added extra validation to ensure special symbols are not present in the dictionary
- Fixed a bug where a trained phonetisaurus model was not properly using its grapheme order
- Fixed a bug when saving a phonetisaurus model after evaluating it

2.0.1
=====

- Fix typo in save model message :github_issue:`470`
- Fix issue with offset alignments when silence words are explicitly in the input transcripts :github_issue:`471`

2.0.0
=====

- Updated and expanded documentation
- Added ability to :ref:`train  Phonetisaurus style G2P models <g2p_phonetisaurus_training>`
- Added support for mixing dictionary formats (i.e., lines can be a mix of non-probabilistic or include pronunciation and silence probabilities)
- Added support for exporting alignments in CSV format
- Updated JSON export format to be more idiomatic JSON :github_issue:`453`
- Fixed a crash where initial training rounds with many jobs would result in jobs that had no utterances :github_issue:`468`
