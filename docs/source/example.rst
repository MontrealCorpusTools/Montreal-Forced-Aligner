.. example:

.. _`LibriSpeech lexicon`: https://drive.google.com/open?id=1dAvxdsHWbtA1ZIh3Ex9DPn9Nemx9M1-L

.. _`LibriSpeech data set`: https://drive.google.com/open?id=1MNlwIv5VyMemrXcZCcC6hENSZpojkdpm

*******
Example
*******

This example for aligning the LibriSpeech test data set assumes that
the Montreal Forced Aligner has been downloaded and works.

Set up
======

1. Download the prepared LibriSpeech dataset (`LibriSpeech data set`_) and extract it somewhere on your computer
2. Download the LibriSpeech lexicon (`LibriSpeech lexicon`_) and save it somewhere on your computer


Alignment
=========

Aligning using pre-trained models
---------------------------------

Enter the following command into the terminal:

.. code-block:: bash

   bin/mfa_align /path/to/librispeech/dataset /path/to/librispeech/lexicon.txt english ~/Documents/aligned_librispeech

Aligning through training
-------------------------

Enter the following command into the terminal:

.. code-block:: bash

   bin/mfa_train_and_align  /path/to/librispeech/dataset /path/to/librispeech/lexicon.txt ~/Documents/aligned_librispeech
