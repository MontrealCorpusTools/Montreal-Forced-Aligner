
.. _`Sigmorphon 2020 G2P task baseline`: https://github.com/sigmorphon/2020/tree/master/task1/baselines/fst

.. _g2p_model_training:

Train a new G2P model ``(mfa train_g2p)``
=========================================

Another tool included with MFA allows you to train a :term:`G2P model` from a given
pronunciation dictionary.
This type of model can be used for :ref:`g2p_dictionary_generating`.
It requires a pronunciation dictionary with each line consisting of the orthographic transcription followed by the
phonetic transcription. The model is generated using the :xref:`pynini` package, which generates FST (finite state transducer)
files. The implementation is based on that in the `Sigmorphon 2020 G2P task baseline`_.
The G2P model output will be a .zip file like the acoustic model generated from alignment.


See :ref:`g2p_model_training_example` for an example of how to train a G2P model with a premade toy example.

.. note::

   As of version 2.0.6, users on Windows can run this command natively without requiring :xref:`wsl`, see :ref:`installation` for more details.

.. _g2p_phonetisaurus_training:

Phonetisaurus style models
--------------------------

As of MFA release 2.0, Phonetisaurus style G2P models are trainable! The default Pynini implementation is based off of a `general pair ngram model <https://github.com/google-research/google-research/tree/master/pair_ngram>`_, however this has the assumption that there is a reasonable one-to-one correspondence between graphemes and phones, with allowances for deletion/insertions covering some one-to-many correspondences.  This works reasonably well for languages that use some form of alphabet, even more non-transparent orthographies like English and French.

However, the basic pair ngram implementation struggles with languages that use syllabaries or logographic systems like Japanese and Chinese.  MFA 1.0 used :xref:`phonetisaurus` as the backend for G2P that had better support for one-to-many mappings.  The Pynini 2.0 implementation encodes strings as paired linear FSAs, so each grapheme leads to the next one, and the model that's being optimized has mappings from all graphemes to all phones and learns their weights.  Phonetisaurus does not encode the input as separate linear FSAs, but rather has single FSTs that cover all the alignments between graphemes and phonemes and then optimizes the FSTs from there.  Phonetisaurus has explicit support for windows of surrounding graphemes and and phonemes, allowing for more efficient learning of patterns like :ipa_inline:`に` :ipa_icon:`right-arrow` :ipa_inline:`[ɲ i]`.

The original Pynini implementation of pair ngram models for G2P was motivated primarily by ease of installation, as Phonetisaurus is not on Conda Forge or has easily installable binaries, so the ``--phonetisaurus`` flag implements the `Phonetisaurus algorithm <https://www.cambridge.org/core/journals/natural-language-engineering/article/abs/phonetisaurus-exploring-graphemetophoneme-conversion-with-joint-ngram-models-in-the-wfst-framework/F1160C3866842F0B707924EB30B8E753>`_ using Pynini.

Command reference
-----------------

.. click:: montreal_forced_aligner.command_line.train_g2p:train_g2p_cli
   :prog: mfa train_g2p
   :nested: full

Configuration reference
-----------------------

- :ref:`configuration_dictionary`
- :ref:`configuration_g2p`

  - :ref:`train_g2p_config`

API reference
-------------

- :ref:`g2p_modeling_api`
