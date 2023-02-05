.. _training_lm:

Train a new language model  ``(mfa train_lm)``
==============================================

MFA has a utility function for training ARPA-format ngram :term:`language models`, as well as merging with a pre-existing model.


.. note::

   As of version 2.0.6, users on Windows can run this command natively without requiring :xref:`wsl`, see :ref:`installation` for more details.

Command reference
-----------------

.. click:: montreal_forced_aligner.command_line.train_lm:train_lm_cli
   :prog: mfa train_lm
   :nested: full

Configuration reference
-----------------------

- :ref:`configuration_language_modeling`

API reference
-------------

- :ref:`language_modeling_api`
