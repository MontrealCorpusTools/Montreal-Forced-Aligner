
.. _pretrained_alignment:

Align with an acoustic model ``(mfa align)``
============================================

This is the primary workflow of MFA, where you can use pretrained :term:`acoustic models` to align your dataset.  There are a number of :ref:`pretrained_acoustic_models` to use, but you can also adapt a pretrained model to your data (see :ref:`adapt_acoustic_model`) or train an acoustic model from scratch using your dataset (see :ref:`train_acoustic_model`).

Command reference
-----------------

.. autoprogram:: montreal_forced_aligner.command_line.mfa:create_parser()
   :prog: mfa
   :start_command: align

Configuration reference
-----------------------

- :ref:`configuration_global`

API reference
-------------

- :ref:`alignment_api`
