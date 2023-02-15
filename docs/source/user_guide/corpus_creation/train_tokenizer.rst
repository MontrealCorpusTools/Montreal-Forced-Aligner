
.. _train_tokenizer_cli:

Train a word tokenizer ``(mfa train_tokenizer)``
================================================

Training a tokenizer uses a simplified sequence-to-sequence model like G2P, but with the following differences:

* Both the input and output symbols are graphemes
* Symbols can only output themselves
* Only allow for inserting space characters

Command reference
-----------------

.. click:: montreal_forced_aligner.command_line.train_tokenizer:train_tokenizer_cli
   :prog: mfa train_tokenizer
   :nested: full


API reference
-------------

- :ref:`tokenization_api`
