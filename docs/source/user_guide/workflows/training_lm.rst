.. _training_lm:

************************
Training language models
************************

MFA has a utility function for training ARPA-format ngram language models, as well as merging with a pre-existing model.


.. warning::

   Please note that this functionality is not available on Windows natively, however, you can install it using :xref:`wsl`, see :ref:`installation_ref` for more details.

Command reference
-----------------

.. autoprogram:: montreal_forced_aligner.command_line.mfa:parser
   :prog: mfa
   :start_command: train_lm
