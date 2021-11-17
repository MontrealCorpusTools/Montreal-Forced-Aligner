.. _classify_speakers:

**********************
Speaker classification
**********************

The Montreal Forced Aligner can use trained ivector models (see :ref:`train_ivector` for more information about training
these models) to classify or cluster utterances according to speakers.

Command reference
-----------------

.. autoprogram:: montreal_forced_aligner.command_line.mfa:parser
   :prog: mfa
   :start_command: classify_speakers
