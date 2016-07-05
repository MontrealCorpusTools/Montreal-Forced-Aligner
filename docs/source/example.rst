.. example:

.. _`LibriSpeech lexicon`: http://www.openslr.org/resources/11/librispeech-lexicon.txt

.. _`LibriSpeech data set`: https://www.dropbox.com/s/i08yunn7yqnbv0h/LibriSpeech.zip?dl=0

*******
Example
*******

This example for aligning the LibriSpeech test data set assumes that
the Montreal Forced Aligner is has been downloaded and works.

Set up
======

1. Download the prepared LibriSpeech dataset (`LibriSpeech data set`_) and extract it somewhere on your computer.
2. Download the LibriSpeech lexicon (`LibriSpeech lexicon`_) and save it somewhere on your computer.


Alignment
=========

Aligning using pre-trained models
---------------------------------

Enter the following command into the terminal:

.. code-block:: bash

   bin/mfa_align /path/to/librispeech/dataset ~/Documents/aligned_librispeech --language english

Aligning through training
-------------------------

Enter the following command into the terminal:

.. code-block:: bash

   bin/mfa_train_and_align  /path/to/librispeech/dataset /path/to/librispeech/lexicon ~/Documents/aligned_librispeech
