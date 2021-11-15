
.. _configuration_dictionary:

************************
Dictionary Configuration
************************

Text normalization and parsing of words from text can be configured in yaml configuration files.  Punctuation is stripped from all words, so if a character is part of a language's orthography, modifying the :code:`punctuation` parameter to exclude that character would keep that character in the words. See more examples of how these :code:`punctuation`, :code:`clitic_markers`, and :code:`compound_markers` are used in :ref:`text_normalization`.

The :code:`multilingual_ipa`, :code:`strip_diacritics`, and :code:`digraphs` are all used as part of :ref:`multilingual_ipa`.

.. csv-table::
   :header: "Parameter", "Default value", "Notes"
   :escape: '

   "oov_word", "<unk>", "Internal word symbol to use for out of vocabulary items"
   "oov_phone", "spn", "Internal phone symbol to use for out of vocabulary items"
   "silence_word", "!sil", "Internal word symbol to use initial silence"
   "nonoptional_silence_phone", "sil", "Internal phone symbol to use initial silence"
   "optional_silence_phone", "sp", "Internal phone symbol to use optional silence in the middle of utterances"
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
