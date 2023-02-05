

.. _`LibriSpeech lexicon`: https://drive.google.com/open?id=1dAvxdsHWbtA1ZIh3Ex9DPn9Nemx9M1-L

.. _`LibriSpeech data set`: https://drive.google.com/open?id=1MNlwIv5VyMemrXcZCcC6hENSZpojkdpm

.. _`THCHS-30`: http://www.openslr.org/18/

.. _`example Mandarin corpus`: https://drive.google.com/file/d/1zPfwvTE_x7o9iX8J8bzeb0KNHEi3jrgN
.. _`example Mandarin dictionary`: https://drive.google.com/file/d/1xCv8-NcAecaUCocNhVRdtSOazE3fjFXf

.. _`Mandarin pinyin G2P model`: http://mlmlab.org/mfa/mfa-models/g2p/mandarin_pinyin_g2p.zip

.. _`Google Colab notebook`: https://gist.github.com/NTT123/12264d15afad861cb897f7a20a01762e

.. _`NTT123`: https://github.com/NTT123

.. _examples:

********
Examples
********

.. _alignment_example:

Example 1: Aligning LibriSpeech (English)
=========================================

.. note::

   There is also a `Google Colab notebook`_ for running the alignment example with a custom Librispeech dataset, created by `NTT123`_.

Set up
------

1. Ensure you have installed MFA via :ref:`installation`.
2. Ensure you have downloaded the pretrained model via :code:`mfa model download acoustic english_mfa`
3. Ensure you have downloaded the pretrained US english dictionary via :code:`mfa model download dictionary english_us_mfa`
4. Download the prepared LibriSpeech dataset (`LibriSpeech data set`_) and extract it somewhere on your computer


Alignment
---------

Aligning using pre-trained models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the same environment that you've installed MFA, enter the following command into the terminal:


.. code-block:: bash

   mfa align /path/to/librispeech/dataset english_us_ma english_mfa ~/Documents/aligned_librispeech

Aligning through training
~~~~~~~~~~~~~~~~~~~~~~~~~

In the same environment that you've installed MFA, enter the following command into the terminal:

.. code-block:: bash

   mfa train  /path/to/librispeech/dataset /path/to/librispeech/lexicon.txt ~/Documents/aligned_librispeech

.. _dict_generating_example:

Example 2: Generate Mandarin dictionary
=======================================

Set up
------

1. Ensure you have installed MFA via :ref:`installation`.
2. Ensure you have downloaded the pretrained model via :code:`mfa model download g2p mandarin_pinyin_g2p`
3. Download the prepared Mandarin dataset from (`example Mandarin corpus`_) and extract it somewhere on your computer

.. note::

   The example Mandarin corpus is .lab files from the `THCHS-30`_ corpus.

To generate a new dictionary for this "corpus" from the pretrained G2P model, run the following:

.. code-block:: bash

   mfa g2p mandarin_pinyin_g2p /path/to/mandarin/dataset /path/to/save/mandarin_dict.txt

This should take no more than a few seconds. Open the output file, and check that all the words are there. The accuracy
of the transcription should be near 100%. You can now use this to align your mini corpus:

.. code-block:: bash

   mfa train /path/to/mandarin/dataset /path/to/save/mandarin_dict.txt /path/to/save/output

Since there are very few files (i.e. small training set), the alignment will be suboptimal. This example is intended more
to give a sense of the pipeline for generating a dictionary and using it for alignment.

.. _g2p_model_training_example:


Example 3: Train Mandarin G2P model
===================================

Set up
------

1. Ensure you have installed MFA via :ref:`installation`.
2. Download the prepared Mandarin dictionary from (`example Mandarin dictionary`_)

In the same environment that you've installed MFA, enter the following command into the terminal:

.. code-block:: bash

    mfa train_g2p /path/to/mandarin_dict.txt mandarin_test_model.zip

This should take no more than a few seconds, and should produce a model which could be used for
:ref:`g2p_dictionary_generating`.

.. note::

   Because there is so little data in ``mandarin_dict.txt``, the model produced will not be very accurate, and so any
   dictionary generated from it will also be inaccurate.  This dictionary is provided for illustrative purposes only.
