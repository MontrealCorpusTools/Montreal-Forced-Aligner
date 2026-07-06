
.. _tokenize_cli:

Tokenize utterances ``(mfa tokenize)``
=========================================

.. deprecated:: 3.4

   The functionality for training tokenizers in MFA is deprecated and slated to be removed in MFA 4.0. For better solutions for tokenizing a given language, see :ref:`language_tokenization` for how to use dedicated packages and models for various languages.

Use a model trained from :ref:`train_tokenizer_cli` to tokenize a corpus (i.e. insert spaces as word boundaries for orthographic systems that do not require them).

Command reference
-----------------

.. click:: montreal_forced_aligner.command_line.tokenize:tokenize_cli
   :prog: mfa tokenize
   :nested: full


API reference
-------------

- :ref:`tokenization_api`
