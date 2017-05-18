.. _api_reference:

*************
API Reference
*************

.. _aligner_api:

Aligner API
===========

There are two main aligner classes, one for using a pretrained model and
one for training a model while aligning.

.. currentmodule:: aligner.aligner

.. autosummary::
   :toctree: generated/
   :template: class.rst

   PretrainedAligner
   TrainableAligner


.. _corpus_api:

Corpus API
==========

The Corpus class contains information about how a dataset is structured

.. currentmodule:: aligner.corpus

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Corpus

.. _dictionary_api:

Dictionary API
==============

.. currentmodule:: aligner.dictionary

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Dictionary

.. _model_api:

Model API
=========

Output from training a model is compressed using the Archive class, which
results in a zip folder.

.. currentmodule:: aligner.models

.. autosummary::
   :toctree: generated/
   :template: class.rst

   AcousticModel
   G2PModel

.. _multiprocessing_api:

Multiprocessing API
===================

The multiprocessing module contains most of the interactions with Kaldi,
as multiple processes are used to speed up the set up and aligning of the
dataset.

.. currentmodule:: aligner.multiprocessing

.. autosummary::
   :toctree: generated/
   :template: function.rst

   mfcc
   compile_train_graphs
   mono_align_equal
   align
   acc_stats
   tree_stats
   calc_fmllr
   convert_alignments
   convert_ali_to_textgrids

Configuration API
=================

These classes contain information about configuring data preparation and
training.

.. currentmodule:: aligner.config

.. autosummary::
   :toctree: generated/
   :template: class.rst

   MfccConfig
   MonophoneConfig
   TriphoneConfig
   TriphoneFmllrConfig

