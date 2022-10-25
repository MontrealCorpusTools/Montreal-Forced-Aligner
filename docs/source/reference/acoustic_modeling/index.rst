
.. _acoustic_modeling_api:

Acoustic models
===============

:term:`Acoustic models` contain information about how phones are pronounced, trained over large (and not-so-large) corpora of speech.  Currently only GMM-HMM style acoustic models are supported, which are generally good enough for alignment, but nowhere near state of the art for transcription.

.. note::

   As part of the training procedure, alignments are generated, and so can be exported at the end (the same as training an acoustic model and then using it with the :class:`~montreal_forced_aligner.alignment.pretrained.PretrainedAligner`. See :meth:`~montreal_forced_aligner.alignment.CorpusAligner.export_files` for the method and :ref:`train_acoustic_model` for the command line function.

.. currentmodule:: montreal_forced_aligner.models

.. autosummary::
  :toctree: generated/

   AcousticModel

.. toctree::
   :hidden:

   training
   helper
