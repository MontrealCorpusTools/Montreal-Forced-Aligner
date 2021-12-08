
.. _configuration_global:

**************
Global Options
**************

These options are used for aligning the full dataset (and as part of training).  Increasing the values of them will
allow for more relaxed restrictions on alignment.  Relaxing these restrictions can be particularly helpful for certain
kinds of files that are quite different from the training dataset (i.e., single word production data from experiments,
or longer stretches of audio).


.. csv-table::
   :header: "Parameter", "Default value", "Notes"
   :escape: '

   "beam", 10, "Initial beam width to use for alignment"
   "retry_beam", 40, "Beam width to use if initial alignment fails"
   "transition_scale", 1.0, "Multiplier to scale transition costs"
   "acoustic_scale", 0.1, "Multiplier to scale acoustic costs"
   "self_loop_scale", 0.1, "Multiplier to scale self loop costs"
   "boost_silence", 1.0, "1.0 is the value that does not affect probabilities"

.. _feature_config:

Feature Configuration
=====================

This section is only relevant for training, as the trained model will contain extractors and feature specification for what they requires.

.. csv-table::
   :header: "Parameter", "Default value", "Notes"

   "feature_type", "mfcc", "Currently only MFCCs are supported"
   "use_energy", "False", "Use energy in place of first MFCC"
   "frame_shift", 10, "In milliseconds, determines time resolution"
   "snip_edges", True, "Should provide better time resolution in alignment"
   "pitch", False, "Currently not implemented"
   "low_frequency", 20, "Frequency cut off for feature generation"
   "high_frequency", 7800, "Frequency cut off for feature generation"
   "sample_frequency", 16000, "Sample rate to up- or down-sample to"
   "allow_downsample", True, "Flag for allowing down-sampling"
   "allow_upsample", True, "Flag for allowing up-sampling"
   "uses_cmvn", True, "Flag for whether to use CMVN"
   "uses_deltas", True, "Flag for whether to use delta features"
   "uses_splices", False, "Flag for whether to use splices and LDA transformations"
   "splice_left_context", 3, "Frame width for generating LDA transforms"
   "splice_right_context", 3, "Frame width for generating LDA transforms"
   "uses_speaker_adaptation", False, "Flag for whether to use speaker adaptation"
   "fmllr_update_type", "full", "Type of fMLLR estimation"
   "silence_weight", 0.0, "Weight of silence in calculating LDA or fMLLR"


.. _configuration_dictionary:

Dictionary and text parsing options
===================================

This sections details configuration options related to how MFA normalizes text and performs dictionary look up.  Punctuation is stripped from all words, so if a character is part of a language's orthography, modifying the :code:`punctuation` parameter to exclude that character would keep that character in the words. See more examples of how these :code:`punctuation`, :code:`clitic_markers`, and :code:`compound_markers` are used in :ref:`text_normalization`.

The :code:`multilingual_ipa`, :code:`strip_diacritics`, and :code:`digraphs` are all used as part of :ref:`multilingual_ipa`.

.. csv-table::
   :header: "Parameter", "Default value", "Notes"
   :escape: '

   "oov_word", "<unk>", "Internal word symbol to use for out of vocabulary items"
   "oov_phone", "spn", "Internal phone symbol to use for out of vocabulary items"
   "silence_word", "!sil", "Internal word symbol to use initial silence"
   "optional_silence_phone", "sp", "Internal phone symbol to use optional silence in or around utterances"
   "position_dependent_phones", "True", "Flag for whether phones should mark their position in the word as part of the phone symbol internally"
   "num_silence_states", "5", "Number of states to use for silence phones"
   "num_non_silence_states", "3", "Number of states to use for non-silence phones"
   "shared_silence_phones", "True", "Flag for whether to share silence phone models"
   "silence_probability", "0.5", "Probability of inserting silence around and within utterances, setting to 0 removes silence modelling"
   "punctuation", "、。।，@<>'"'(),.:;¿?¡!\\&%#*~【】，…‥「」『』〝〟″⟨⟩♪・‹›«»～′$+=", "Characters to treat as punctuation and strip from around words"
   "clitic_markers", "'''’", "Characters to treat as clitic markers, will be collapsed to the first character in the string"
   "compound_markers", "\-", "Characters to treat as marker in compound words (i.e., doesn't need to be preserved like for clitics)"
   "multilingual_ipa", False, "Flag for enabling multilingual IPA mode, see :ref:`multilingual_ipa` for more details"
   "strip_diacritics", "/iː/ /iˑ/ /ĭ/ /i̯/  /t͡s/ /t‿s/ /t͜s/ /n̩/", "IPA diacritics to strip in multilingual IPA mode (phone symbols for proper display, when specifying them just have the diacritic)"
   "digraphs", "[dt][szʒʃʐʑʂɕç], [aoɔe][ʊɪ]", "Digraphs to split up in multilingual IPA mode"
   "brackets", "('[', ']'), ('{', '}'), ('<', '>'), ('(', ')')", "Punctuation to keep as bracketing a whole word, i.e., a restart, disfluency, etc"
