.. _training_dictionary:

Add probabilities to a dictionary ``(mfa train_dictionary)``
============================================================

MFA includes a utility command for training :term:`pronunciation probabilities` of a dictionary given a corpus for alignment.

The resulting dictionary can then be used as a dictionary for alignment or transcription.


Command reference
-----------------

.. autoprogram:: montreal_forced_aligner.command_line.mfa:create_parser()
   :prog: mfa
   :start_command: train_dictionary
