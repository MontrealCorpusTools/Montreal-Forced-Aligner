.. _adapt_acoustic_model:

***********************************
Adapting acoustic model to new data
***********************************

A recent 2.0 functionality for MFA is to adapt pretrained models to a new dataset.  MFA will first align the dataset using the pretrained model, and then perform a couple of rounds of speaker-adaptation training.


Command reference
-----------------

.. autoprogram:: montreal_forced_aligner.command_line.mfa:parser
   :prog: mfa
   :start_command: adapt
