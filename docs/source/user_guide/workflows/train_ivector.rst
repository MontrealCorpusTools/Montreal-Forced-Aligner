.. _train_ivector:

*****************************
Training an ivector extractor
*****************************

The Montreal Forced Aligner can train ivector extractors using an acoustic model for generating alignments.  As part
of this training process, a classifier is built in that can be used as part of :ref:`classify_speakers`.

Command reference
-----------------

.. autoprogram:: montreal_forced_aligner.command_line.mfa:parser
   :prog: mfa
   :start_command: train_ivector
