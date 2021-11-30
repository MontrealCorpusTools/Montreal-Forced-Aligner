.. _mfa_api:

MFA API
=======

.. warning::

   While the MFA command-line interface is fairly stable, I do tend to do refactors of the internal code on fairly regular basis.  As 2.0 gets more stable, these are likely to get smaller and smaller, and I will try to keep the API docs as up-to-date as possible, so if something breaks in any scripts depending on MFA, please check back here.

Current structure
-----------------

Prior to 2.0.0b8, MFA classes were fairly monolithic.  There was a ``Corpus`` class that did everything related to corpus loading and processing text and sound files.  However, the default acoustic model with sound files for alignment does not necessarily lend itself to language modeling for instance, and so there were several flags for text-only behavior that didn't feel satisfying.

A bigger concern was as more configuration options were added to for processing pronunciation dictionaries, they would have to be duplicated in the existing workflow configuration objects (AlignConfig, TranscribeConfig, etc), or a new DictionaryConfig object that gets passed to all workflow classes (PretrainedAligner, Transcriber, etc) and data processing classes (Corpus, Dictionary).

The current design mixes in functionality as necessary with abstract classes.  So there is a :class:`~montreal_forced_aligner.dictionary.mixins.DictionaryMixin` class that covers the functionality around what counts as a word, how to parse text through stripping punctuation, using compound and clitic markers in looking up words, etc.  There are several :class:`~montreal_forced_aligner.corpus.base.CorpusMixin` classes that have similar data structure and attributes, but different load functionality for corpora with sound files (:class:`~montreal_forced_aligner.corpus.acoustic_corpus.AcousticCorpusMixin` and :class:`~montreal_forced_aligner.corpus.acoustic_corpus.AcousticCorpusPronunciationMixin`, depending on whether a pronunciation dictionary is needed for processing) versus text-only corpora (:class:`~montreal_forced_aligner.corpus.text_corpus.TextCorpusMixin` and :class:`~montreal_forced_aligner.corpus.text_corpus.DictionaryTextCorpusMixin`).

This should (hopefully) make it easier to extend MFA for your own purposes if you so choose, and will certainly make it easier for me to implement new functionality going forward.

.. toctree::
   :hidden:

   core_index
   top_level_index
   helper/index
