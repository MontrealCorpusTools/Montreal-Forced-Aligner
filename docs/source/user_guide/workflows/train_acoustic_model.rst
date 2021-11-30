.. _train_acoustic_model:

Train a new acoustic model ``(mfa train)``
==========================================

You can train new :term:`acoustic models` from scratch using MFA, and export the final alignments as :term:`TextGrids` at the end.  You don't need a ton of data to generate decent alignments (see `the blog post comparing alignments trained on various corpus sizes <https://memcauliffe.com/how-much-data-do-you-need-for-a-good-mfa-alignment.html>`_).  At the end of the day, it comes down to trial and error, so I would recommend trying different workflows of pretrained models vs training your own or adapting a model to your data to see what performs best.


Command reference
-----------------


.. autoprogram:: montreal_forced_aligner.command_line.mfa:create_parser()
   :prog: mfa
   :start_command: train

Configuration reference
-----------------------

- :ref:`configuration_acoustic_modeling`

API reference
-------------

- :ref:`acoustic_modeling_api`

  - :ref:`acoustic_model_training_api`
