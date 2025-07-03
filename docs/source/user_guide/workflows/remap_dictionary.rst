.. _remap_dictionary:

Remap a dictionary to new phone set ``(mfa remap_dictionary)``
==============================================================

If you have a mismatch in the phone sets used in your dictionary file and acoustic model, you can use this command to command to generate pronunciations with the new phone set.

Command reference
-----------------

.. click:: montreal_forced_aligner.command_line.remap_dictionary:remap_dictionary_cli
   :prog: mfa remap_dictionary
   :nested: full

Configuration reference
-----------------------

- :ref:`configuration_global`

API reference
-------------

- :class:`~montreal_forced_aligner.dictionary.remapper.DictionaryRemapper`
