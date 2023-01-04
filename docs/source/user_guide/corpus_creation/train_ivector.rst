.. _train_ivector:

Train an ivector extractor ``(mfa train_ivector)``
==================================================

The Montreal Forced Aligner can train :term:`ivector extractors` using an acoustic model for generating alignments.  As part of this training process, a classifier is built in that can be used as part of :ref:`diarize_speakers`.

.. warning::

   This feature is not fully implemented, and is still under construction.

Command reference
-----------------

.. click:: montreal_forced_aligner.command_line.train_ivector_extractor:train_ivector_cli
   :prog: mfa train_ivector
   :nested: full

Configuration reference
-----------------------

- :ref:`configuration_ivector`

API reference
-------------

- :ref:`ivector_api`
