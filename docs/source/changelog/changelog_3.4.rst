
.. _changelog_3.4:

*************
3.4 Changelog
*************

3.4.0
-----

- Compatibility with Kalpy 0.9.0
- Introduced new command for :ref:`compare_alignments`
- Introduced new MFA model format for distribution on HuggingFace
- Introduced new commands that use the new MFA model format (i.e. :ref:`pretrained_alignment_hf`) which will replace the default commands (i.e. :ref:`pretrained_alignment`) in MFA 4.0.  The default commands will still be available using a legacy command and will be removed in MFA 5.0.
- Deprecated :ref:`train_tokenizer_cli` and :ref:`tokenize_cli` for removal in MFA 4.0
- Added calculation for :ref:`alignment_analysis_intensity_deviation` and :ref:`alignment_analysis_snr` to be included as diagnostic metrics for alignment
