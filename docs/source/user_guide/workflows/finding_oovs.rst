

.. _g2p_find_oovs:

Find OOVs in a corpus ``(mfa find_oovs)``
=========================================

The ``mfa find_oovs`` command is a utility for generating a list of OOVs for a given corpus and pronunciation dictionary, along with counts of their occurrences in the corpus and which utterances they appear in.

.. note::

   This command is functionally the same as :ref:`using the corpus validator <running_the_validator>`, but it outputs the OOV information more straight-forwardly.


Command reference
-----------------

.. click:: montreal_forced_aligner.command_line.find_oovs:find_oovs_cli
   :prog: mfa find_oovs
   :nested: full
