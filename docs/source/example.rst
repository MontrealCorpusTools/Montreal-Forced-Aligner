

.. _`LibriSpeech lexicon`: http://www.openslr.org/resources/11/librispeech-lexicon.txt

.. _`LibriSpeech data set`: https://www.dropbox.com/s/i08yunn7yqnbv0h/LibriSpeech.zip?dl=0

.. _`THCHS-30`: http://www.openslr.org/18/

.. _`example Mandarin corpus`: http://mlmlab.org/mfa/CH_g2p_example.zip

.. _`Mandarin pinyin G2P model`: http://mlmlab.org/mfa/mfa-models/g2p/mandarin_pinyin_g2p.zip

.. _examples:

********
Examples
********

.. _alignment_example:

Example 1: Aligning LibriSpeech (English)
=========================================

Set up
------

1. Download the prepared LibriSpeech dataset (`LibriSpeech data set`_) and extract it somewhere on your computer
2. Download the LibriSpeech lexicon (`LibriSpeech lexicon`_) and save it somewhere on your computer


Alignment
---------

Aligning using pre-trained models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the root directory of the Montreal Forced Aligner, enter the following command into the terminal:

.. code-block:: bash

   bin/mfa_align /path/to/librispeech/dataset /path/to/librispeech/lexicon.txt english ~/Documents/aligned_librispeech

Aligning through training
~~~~~~~~~~~~~~~~~~~~~~~~~

From the root directory of the Montreal Forced Aligner, enter the following command into the terminal:

.. code-block:: bash

   bin/mfa_train_and_align  /path/to/librispeech/dataset /path/to/librispeech/lexicon.txt ~/Documents/aligned_librispeech

.. _dict_generating_example:

Example 2: Generate Mandarin dictionary
=======================================

Download the `example Mandarin corpus`_ and the `Mandarin pinyin G2P model`_ to some place on your machine. In ``examples/CH`` you will find several sample .lab files (orthographic transcriptions)
from the `THCHS-30`_ corpus. These are organized much as they would be for any alignment task. The dictionary reconstructor will
create a word list of all the orthographic word-forms in the files, and will build a pronunciation dictionary with a
phonetic transcription for each one of these words, which it will write to a file. Let's start by running the reconstructor, as before:

.. code-block:: bash

   bin/mfa_generate_dictionary /path/to/mandarin_pinyin_g2p.zip /path/to/examples/CH /path/to/examples/CH chinese_dict.txt

This should take no more than a few seconds. Open the output file, and check that all the words are there. The accuracy
of the transcription should be near 100%. You can now use this to align your mini corpus:

.. code-block:: bash

   bin/mfa_train_and_align path/to/examples/CH  path/to/examples/chinese_dict.txt examples/aligned_output

Since there are very few files (i.e. small training set), the alignment will be suboptimal. This example is intended more
to give a sense of the pipeline for generating a dictionary and using it for alignment.

.. _g2p_model_training_example:


Example 3: Train Mandarin G2P model
===================================

Download the `example Mandarin corpus`_ to some place on your machine.
In the ``examples`` folder, you will find a small Chinese dictionary (``chinese_dict.txt``). It is too small to generate a usable model, but can provide a helpful example. Inputting

.. code-block:: bash

    bin/mfa_train_g2p /path/to/examples/chinese_dict.txt CH_test_model.zip

This should take no more than a few seconds, and should produce a model which could be used for :doc:`generating dictionaries <dictionary_generating>`

.. note::

   Because there is so little data in chinese_dict.txt, the model produced will not be very accurate, and so any
   dictionary generated from it will also be inaccurate.