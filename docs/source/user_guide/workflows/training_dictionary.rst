.. _training_dictionary:

************************************
Modeling pronunciation probabilities
************************************

MFA includes a utility command for training pronunciation probabilities of a dictionary given a corpus for alignment.

The resulting dictionary can then be used as a dictionary for alignment or transcription.


Command reference
-----------------

.. autoprogram:: montreal_forced_aligner.command_line.mfa:parser
   :prog: mfa
   :start_command: train_dictionary
