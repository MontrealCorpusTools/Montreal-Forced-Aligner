
.. _changelog_3.1:

*************
3.1 Changelog
*************

3.1.4
-----

- Optimized :code:`mfa g2p` to better use multiple processes
- Added :code:`--export_scores` to :code:`mfa g2p` for adding a column representing the final weights of the generated pronunciations
- Added :code:`--output_directory` to :code:`mfa validate` to save generated validation files rather than the temporary directory
- Fixed a bug in cutoff modeling that was preventing them from being properly parsed

3.1.3
-----

- Fixed an issue where silence probability being zero was not correctly removing silence
- Compatibility with kalpy v0.6.5
- Added API functionality for verifying transcripts with interjection words in alignment
- Fixed an error in fine tuning that generated nonsensical boundaries

3.1.2
-----

- Fixed a bug where hidden files and folders would be parsed as corpus data
- Fixed a bug where validation would not respect :code:`--no_final_clean`
- Fixed a rare crash in training when a job would not have utterances assigned to it
- Fixed a bug where MFA would mistakenly report a dictionary and acoustic model phones did not match for older versions

3.1.1
-----

- Fixed an issue with TextGrids missing intervals

3.1.0
-----

- Fixed a bug where cutoffs were not properly modelled
- Added additional filter on create subset to not include utterances with cutoffs in smaller subsets
- Added the ability to specify HMM topologies for phones
- Fixed issues caused by validators not cleaning up temporary files and databases
- Added support for default and nonnative dictionaries generated from other dictionaries
- Restricted initial training rounds to exclude default and nonnative dictionaries
- Changed clustering of phones to not mix silence and non-silence phones
- Optimized textgrid export
- Added better memory management for collecting alignments
