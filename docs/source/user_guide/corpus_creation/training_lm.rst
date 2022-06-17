.. _training_lm:

Train a new language model  ``(mfa train_lm)``
==============================================

MFA has a utility function for training ARPA-format ngram :term:`language models`, as well as merging with a pre-existing model.


.. warning::

   Please note that this functionality is not available on Windows natively, however, you can install it using :xref:`wsl`, see :ref:`installation` for more details.

Command reference
-----------------

.. autoprogram:: montreal_forced_aligner.command_line.mfa:create_parser()
   :prog: mfa
   :start_command: train_lm

Configuration reference
-----------------------

- :ref:`configuration_language_modeling`

API reference
-------------

- :ref:`language_modeling_api`
