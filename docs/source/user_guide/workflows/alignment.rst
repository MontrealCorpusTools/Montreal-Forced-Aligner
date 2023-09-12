
.. _pretrained_alignment:

Align with an acoustic model ``(mfa align)``
============================================

This is the primary workflow of MFA, where you can use pretrained :term:`acoustic models` to align your dataset.  There are a number of :xref:`pretrained_acoustic_models` to use, but you can also adapt a pretrained model to your data (see :ref:`adapt_acoustic_model`) or train an acoustic model from scratch using your dataset (see :ref:`train_acoustic_model`).

.. seealso::

   * :ref:`alignment_evaluation` for details on how to evaluate alignments against a gold standard.
   * :ref:`fine_tune_alignments` for implementation details on how alignments are fine tuned.
   * :ref:`phone_models` for implementation details on using phone bigram models for generating alignments.
   * :ref:`alignment_analysis` for details on the fields generated in the ``alignment_analysis.csv`` file in the output folder

Command reference
-----------------

.. click:: montreal_forced_aligner.command_line.align:align_corpus_cli
   :prog: mfa align
   :nested: full

Configuration reference
-----------------------

By default, the acoustic model controls parameters related to silence probability or speaker adaptation.  These can be overridden in the command line so `--initial_silence_probability 0.0` will ensure that no utterances start with silence, and `--uses_speaker_adaptation false` will skip the feature space adaptation and second pass alignment.

.. seealso::

   See :ref:`concept_speaker_adaptation` for more details on how speaker adaptation works in Kaldi/MFA.

- :ref:`configuration_global`

API reference
-------------

- :ref:`alignment_api`

.. _align_one:

Align a single file ``(mfa align_one)``
=======================================

This workflow is identical to :ref:`pretrained_alignment`, but rather than aligning a full dataset, it only aligns a single file.
Because only a single file is used, many of the optimizations for larger datasets are skipped resulting in faster alignment times,
but features like speaker adaptation are not employed.

There are a number of :xref:`pretrained_acoustic_models` to use, but you can also adapt a pretrained model to your data (see :ref:`adapt_acoustic_model`)
or train an acoustic model from scratch using your dataset (see :ref:`train_acoustic_model`).

Command reference
-----------------

.. click:: montreal_forced_aligner.command_line.align_one:align_one_cli
   :prog: mfa align_one
   :nested: full

Configuration reference
-----------------------

- :ref:`configuration_global`
