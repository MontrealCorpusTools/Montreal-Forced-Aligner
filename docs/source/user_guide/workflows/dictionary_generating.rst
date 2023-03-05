

.. _g2p_dictionary_generating:

Generate a new pronunciation dictionary ``(mfa g2p)``
=====================================================

We have trained several G2P models that are available for download (:xref:`pretrained_g2p`).

.. warning::

   Please note that G2P models trained prior to 2.0 cannot be used with MFA 2.0.  If you would like to use
   these models, please use the the 1.0.1 or 1.1 g2p utilities or retrain a new G2P model following
   :ref:`g2p_model_training`.

.. note::

   Generating pronunciations to supplement your existing pronunciation
   dictionary can be done by running the validation utility (see :ref:`running_the_validator`), and then use the path
   to the ``oovs_found.txt`` file that it generates.


Pronunciation dictionaries can also be generated from the orthographies of the words themselves, rather than relying on
a trained G2P model.  This functionality should be reserved for languages with transparent orthographies, close to 1-to-1
grapheme-to-phoneme mapping.

See :ref:`dict_generating_example` for an example of how to use G2P functionality with a premade example.

.. note::

   As of version 2.0.6, users on Windows can run this command natively without requiring :xref:`wsl`, see :ref:`installation` for more details.

Piping stdin/stdout
-------------------

If you specify the input path as ``-`` instead of a file path, the g2p command will run through each line in the stdin and G2P each word with minimal processing.  Words will be lower cased and any graphemes that were not in the model's training data will be removed.

If you specify the output path as ``-`` instead of a file path, the g2p command will send pronunciations as stdout rather than writing to a file.

.. note::

   Using stdin will also bypass database set up (though the database server will still be started and stopped, so be sure to run :code:`mfa configure --no_auto_server` if speed is of necessity.

Command reference
-----------------

.. click:: montreal_forced_aligner.command_line.g2p:g2p_cli
   :prog: mfa g2p
   :nested: full

Configuration reference
-----------------------

- :ref:`configuration_g2p`
- :ref:`configuration_dictionary`

API reference
-------------

- :ref:`g2p_generate_api`
