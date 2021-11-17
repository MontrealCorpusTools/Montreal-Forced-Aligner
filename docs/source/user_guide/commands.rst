

.. _commands:

************
All commands
************

The ``mfa`` command line utility has several subcommands, which are listed below grouped by general domain.

Preparation
===========

.. csv-table::
   :header: "Command", "Description", "Link"
   :widths: 10, 110, 40

   "validate", "Validate a corpus", :ref:`validating_data`

Forced Alignment
================

.. csv-table::
   :header: "Command", "Description", "Link"
   :widths: 10, 110, 40

   "align", "Perform forced alignment with a pretrained model", :ref:`pretrained_alignment`
   "train", "Train an acoustic model and export resulting alignment", :ref:`train_acoustic_model`
   "adapt", "Adapt a pretrained acoustic model on a new dataset", :ref:`adapt_acoustic_model`
   "train_dictionary", "Estimate pronunciation probabilities from aligning a corpus", :ref:`training_dictionary`

Corpus creation
===============

.. csv-table::
   :header: "Command", "Description", "Link"
   :widths: 10, 110, 40

   "create_segments", "Use voice activity detection to create segments", :ref:`create_segments`
   "train_ivector", "Train an ivector extractor for speaker classification", :ref:`train_ivector`
   "classify_speakers", "Use ivector extractor to classify files or cluster them", :ref:`classify_speakers`
   "transcribe", "Generate transcriptions using an acoustic model, dictionary, and language model", :ref:`transcribing`
   "train_lm", "Train a language model from a text corpus or from an existing language model", :ref:`training_lm`
   "anchor", "Run the Anchor annotator utility (if installed) for editing and managing corpora", :ref:`anchor`

Other utilities
===============

.. csv-table::
   :header: "Command", "Description", "Link"
   :widths: 10, 110, 40

   "model", "Inspect/list/download/save models", :ref:`pretrained_models`
   "configure", "Configure MFA to use customized defaults for command line arguments", :ref:`configuration`
   "history", "List previous MFA commands run locally",


Grapheme-to-phoneme
===================

.. csv-table::
   :header: "Command", "Description", "Link"
   :widths: 10, 110, 40

   "g2p", "Use a G2P model to generate a pronunciation dictionary", :ref:`g2p_dictionary_generating`
   "train_g2p", "Train a G2P model from a pronunciation dictionary", :ref:`g2p_model_training`
