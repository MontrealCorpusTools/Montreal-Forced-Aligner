

.. _commands:

************
All commands
************

The ``mfa`` command line utility has several subcommands, which are listed below grouped by general domain.

Preparation
===========

.. csv-table::
   :header: "Command", "Description", "Link"
   :widths: 50, 110, 40

   "``mfa validate``", "Validate a corpus", :ref:`validating_data`

Forced Alignment
================

.. csv-table::
   :header: "Command", "Description", "Link"
   :widths: 50, 110, 40

   "``mfa align``", "Perform forced alignment with a pretrained model", :ref:`pretrained_alignment`
   "``mfa train``", "Train an acoustic model and export resulting alignment", :ref:`train_acoustic_model`
   "``mfa adapt``", "Adapt a pretrained acoustic model on a new dataset", :ref:`adapt_acoustic_model`
   "``mfa train_dictionary``", "Estimate pronunciation probabilities from aligning a corpus", :ref:`training_dictionary`

Corpus creation
===============

.. csv-table::
   :header: "Command", "Description", "Link"
   :widths: 50, 110, 40

   "``mfa create_segments``", "Use voice activity detection to create segments", :ref:`create_segments`
   "``mfa train_ivector``", "Train an ivector extractor for speaker classification", :ref:`train_ivector`
   "``mfa diarize_speakers``", "Use ivector extractor to classify files or cluster them", :ref:`diarize_speakers`
   "``mfa transcribe``", "Generate transcriptions using an acoustic model, dictionary, and language model", :ref:`transcribing`
   "``mfa train_lm``", "Train a language model from a text corpus or from an existing language model", :ref:`training_lm`
   "``mfa anchor``", "Run the Anchor annotator utility (if installed) for editing and managing corpora", :ref:`anchor`

Other utilities
===============

.. csv-table::
   :header: "Command", "Description", "Link"
   :widths: 50, 110, 40

   "``mfa model``", "Inspect/list/download/save models", :ref:`pretrained_models`
   "``mfa configure``", "Configure MFA to use customized defaults for command line arguments", :ref:`configuration`
   "``mfa history``", "List previous MFA commands run locally",


Grapheme-to-phoneme
===================

.. csv-table::
   :header: "Command", "Description", "Link"
   :widths: 50, 110, 40

   "``mfa g2p``", "Use a G2P model to generate a pronunciation dictionary", :ref:`g2p_dictionary_generating`
   "``mfa train_g2p``", "Train a G2P model from a pronunciation dictionary", :ref:`g2p_model_training`
