
.. _`Sigmorphon 2020 G2P task baseline`: https://github.com/sigmorphon/2020/tree/master/task1/baselines/fst

.. _g2p_model_training:

Train a new G2P model ``(mfa train_g2p)``
=========================================

Another tool included with MFA allows you to train a :term:`G2P model` from a given
pronunciation dictionary.
This type of model can be used for :ref:`g2p_dictionary_generating`.
It requires a pronunciation dictionary with each line consisting of the orthographic transcription followed by the
phonetic transcription. The model is generated using the :xref:`pynini` package, which generates FST (finite state transducer)
files. The implementation is based on that in the `Sigmorphon 2020 G2P task baseline`_.
The G2P model output will be a .zip file like the acoustic model generated from alignment.


See :ref:`g2p_model_training_example` for an example of how to train a G2P model with a premade toy example.

.. warning::

   Please note that this functionality is not available on Windows natively, however, you can install it using :xref:`wsl`, see :ref:`installation` for more details.

Command reference
-----------------

.. autoprogram:: montreal_forced_aligner.command_line.mfa:create_parser()
   :prog: mfa
   :start_command: train_g2p

Configuration reference
-----------------------

- :ref:`configuration_dictionary`
- :ref:`configuration_g2p`

  - :ref:`train_g2p_config`

API reference
-------------

- :ref:`g2p_modeling_api`
