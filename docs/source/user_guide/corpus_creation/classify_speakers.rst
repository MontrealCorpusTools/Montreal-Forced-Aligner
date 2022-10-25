.. _classify_speakers:

Cluster speakers ``(mfa classify_speakers)``
============================================

The Montreal Forced Aligner can use trained ivector models (see :ref:`train_ivector` for more information about trainingthese models) to classify or cluster utterances according to speakers.

.. warning::

   This feature is not fully implemented, and is still under construction.

Command reference
-----------------

.. click:: montreal_forced_aligner.command_line.classify_speakers:classify_speakers_cli
   :prog: mfa classify_speakers
   :nested: full
