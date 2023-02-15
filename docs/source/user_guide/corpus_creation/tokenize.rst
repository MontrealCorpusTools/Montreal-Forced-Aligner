
.. _tokenize_cli:

Tokenize utterances ``(mfa tokenize)``
=========================================

Use a model trained from :ref:`train_tokenizer_cli` to tokenize a corpus (i.e. insert spaces as word boundaries for orthographic systems that do not require them).

Command reference
-----------------

.. click:: montreal_forced_aligner.command_line.tokenize:tokenize_cli
   :prog: mfa tokenize
   :nested: full


API reference
-------------

- :ref:`tokenization_api`
