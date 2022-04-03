

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

.. warning::

   Please note that this functionality is not available on Windows natively, however, you can install it using :xref:`wsl`, see :ref:`installation` for more details.

Command reference
-----------------

.. autoprogram:: montreal_forced_aligner.command_line.mfa:create_parser()
   :prog: mfa
   :start_command: g2p

Configuration reference
-----------------------

- :ref:`configuration_g2p`
- :ref:`configuration_dictionary`

API reference
-------------

- :ref:`g2p_generate_api`
