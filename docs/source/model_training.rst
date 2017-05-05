.. _model_training:

.. _`THCHS-30`: http://www.openslr.org/18/
.. _`Phonetisaurus`: https://github.com/AdolfVonKleist/Phonetisaurus



************************
Training a new G2P model
************************

Another tool included with MFA allows you to train a G2P (Grapheme to Phoneme) model automatically from a given pronunciation dictionary.
This type of model can be used for :doc:`generating dictionaries <dictionary_generating>`.
It requires a pronunciation dictionary with each line consisting of the orthographic transcription followed by the
phonetic transcription. The model is generated using the `Phonetisaurus`_ software, which generates FST (finite state transducer)
files. The G2P model output will be a .zip file like the acoustic model generated from alignment.

Use
===

To train a model from a pronunciation dictionary, the following command is used:

.. code-block:: bash

    bin/mfa_train_g2p /path/to/dictionary/file /path/to/output/dictionary

Extra options:

.. cmdoption:: --window_size NUM_PHONES

   This should be used if there are instances of a single orthographic
   character corresponding to more than 2 phones (common in Korean hangul, Chinese character orthography, etc.).


Example
=======

In ```Montreal-Forced-Aligner/examples/``` you will find a small Chinese dictionary. It is too small to generate a usable model, but can provide a helpful example. Inputting 

.. code-block:: bash

    bin/mfa_train_g2p Montreal-Forced-Aligner/examples/chinese_dict.txt Montreal-Forced-Aligner/examples/CH_test_model

This should take no more than a few seconds, and should produce a model which could be used for :doc:`generating dictionaries <dictionary_generating>` 

.. note::

   Because there is so little data in chinese_dict.txt, the model produced will not be very accurate. Thus any dictionary generated from it will also be inaccurate.
