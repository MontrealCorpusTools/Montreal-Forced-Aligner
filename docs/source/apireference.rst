.. _api_reference:

*************
API Reference
*************

Below is a diagram of the main classes used in MFA:

.. .. image:: ../build/html/_images/generalUML.svg

.. _aligner_api:

Aligner API
===========

There are two main Aligner classes, one for using a pretrained model and
one for training a model while aligning. A class diagram of the Aligner API can be found below:


..
  .. image:: ../build/html/_images/alignerUML.svg
         :height: 200 px
         :width: 300 px
         :align: center

.. currentmodule:: aligner.aligner

.. autosummary::
   :toctree: generated/
   :template: class.rst

   PretrainedAligner
   TrainableAligner


.. _corpus_api:

Corpus API
==========

The Corpus class contains information about how a dataset is structured. A class diagram of the Corpus API can be found below:

..
  .. image:: ../build/html/_images/corpusUML.svg
         :height: 200 px
         :width: 300 px
         :align: center

.. currentmodule:: aligner.corpus

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Corpus

.. _dictionary_api:

Dictionary API
==============

The Dictionary class contains pronunciation and orthographic information. A class diagram of the Dictionary API can be found below:

..
  .. image:: ../build/html/_images/dictionaryUML.svg
         :height: 200 px
         :width: 300 px
         :align: center

.. currentmodule:: aligner.dictionary

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Dictionary

.. _model_api:

Model API
=========

Output from training a Model is compressed using the Archive class, which
results in a zip folder. A class diagram of the Model API can be found below:

..
  .. image:: ../build/html/_images/modelUML.svg
         :height: 400 px
         :width: 500 px
         :align: center

.. currentmodule:: aligner.models

.. autosummary::
   :toctree: generated/
   :template: class.rst

   AcousticModel
   G2PModel
   IvectorExtractor

.. _feature_processing_api:

Feature processing API
======================

.. currentmodule:: aligner.features.processing

.. autosummary::
   :toctree: generated/
   :template: function.rst

   mfcc
   apply_cmvn
   add_deltas
   apply_lda

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

   compile_train_graphs
   mono_align_equal
   align
   acc_stats
   tree_stats
   calc_fmllr
   convert_alignments
   convert_ali_to_textgrids

For use with DNNs
-----------------

.. currentmodule:: aligner.multiprocessing

.. autosummary::
   :toctree: generated/
   :template: function.rst

   lda_acc_stats
   calc_lda_mllt
   gmm_gselect
   acc_global_stats
   gauss_to_post
   acc_ivector_stats
   extract_ivectors
   get_egs
   get_lda_nnet
   nnet_train_trans
   nnet_train
   nnet_align
   compute_prob
   get_average_posteriors
   relabel_egs

Trainer API
===========

These Trainer classes contain information about configuring data preparation and
training. A class diagram of the Configuration API can be found below:

..
  .. image:: ../build/html/_images/configUML.svg
         :height: 600 px
         :width: 800 px
         :align: center

.. currentmodule:: aligner.trainers

.. autosummary::
   :toctree: generated/
   :template: class.rst

   MonophoneTrainer
   TriphoneTrainer
   LdaTrainer
   SatTrainer
   IvectorExtractorTrainer
   NnetTrainer

