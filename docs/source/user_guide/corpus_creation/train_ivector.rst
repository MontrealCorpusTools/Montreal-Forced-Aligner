.. _train_ivector:

Train an ivector extractor ``(mfa train_ivector)``
==================================================

The Montreal Forced Aligner can train :term:`ivector extractors` using an acoustic model for generating alignments.  As part of this training process, a classifier is built in that can be used as part of :ref:`classify_speakers`.

.. warning::

   This feature is not fully implemented, and is still under construction.

Command reference
-----------------

.. autoprogram:: montreal_forced_aligner.command_line.mfa:create_parser()
   :prog: mfa
   :start_command: train_ivector

Configuration reference
-----------------------

- :ref:`configuration_ivector`

API reference
-------------

- :ref:`ivector_api`
