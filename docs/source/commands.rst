.. _commands:

********
Commands
********

The ``mfa`` command line utility has several subcommands, which are listed below grouped by general domain.

Forced Alignment
================

.. csv-table::
   :header: "Command", "Description", "Link"
   :widths: 10, 110, 40

   "align", "Perform forced alignment with a pretrained model", :ref:`pretrained_alignment`
   "train", "Train an acoustic model and export resulting alignment", :ref:`trained_alignment`
   "adapt", "Adapt a pretrained acoustic model on a new dataset", :ref:`adapting_model`
   "validate", "Validate a corpus to ensure there are no issues with the data format", :ref:`validating_data`
   "train_dictionary", "Estimate pronunciation probabilities from aligning a corpus", :ref:`training_dictionary`


Transcription
=============

.. csv-table::
   :header: "Command", "Description", "Link"
   :widths: 10, 110, 40

   "transcribe", "Generate transcriptions using an acoustic model, dictionary, and language model", :ref:`transcribing`
   "train_lm", "Train a language model from a text corpus or from an existing language model", :ref:`training_lm`

Corpus creation
===============

.. csv-table::
   :header: "Command", "Description", "Link"
   :widths: 10, 110, 40

   "create_segments", "Use voice activity detection to create segments", :ref:`create_segments`
   "train_ivector", "Train an ivector extractor for speaker classification", :ref:`train_ivector`
   "classify_speakers", "Use ivector extractor to classify files or cluster them", :ref:`classify_speakers`
   "annotator", "Run a GUI annotator program for editing and managing corpora", :ref:`annotator`


Other utilities
===============

.. csv-table::
   :header: "Command", "Description", "Link"
   :widths: 10, 110, 40

   "download", "Download a model trained by MFA developers", :ref:`pretrained_models`
   "thirdparty", "Download and validate new third party binaries", :ref:`installation`
   "configure", "Configure MFA to use customized defaults for command line arguments", :ref:`installation`


Grapheme-to-phoneme
===================

.. csv-table::
   :header: "Command", "Description", "Link"
   :widths: 10, 110, 40

   "g2p", "Use a G2P model to generate a pronunciation dictionary", :ref:`g2p_dictionary_generating`
   "train_g2p", "Train a G2P model from a pronunciation dictionary", :ref:`g2p_model_training`
