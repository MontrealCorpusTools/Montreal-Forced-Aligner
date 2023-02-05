
.. _pretrained_alignment:

Align with an acoustic model ``(mfa align)``
============================================

This is the primary workflow of MFA, where you can use pretrained :term:`acoustic models` to align your dataset.  There are a number of :xref:`pretrained_acoustic_models` to use, but you can also adapt a pretrained model to your data (see :ref:`adapt_acoustic_model`) or train an acoustic model from scratch using your dataset (see :ref:`train_acoustic_model`).

.. seealso::

   * :ref:`alignment_evaluation` for details on how to evaluate alignments against a gold standard.
   * :ref:`fine_tune_alignments` for implementation details on how alignments are fine tuned.
   * :ref:`phone_models` for implementation details on using phone bigram models for generating alignments.

Command reference
-----------------

.. click:: montreal_forced_aligner.command_line.align:align_corpus_cli
   :prog: mfa align
   :nested: full

Configuration reference
-----------------------

- :ref:`configuration_global`

API reference
-------------

- :ref:`alignment_api`
