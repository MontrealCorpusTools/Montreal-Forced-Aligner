.. example:

.. _`LibriSpeech lexicon`: http://www.openslr.org/resources/11/librispeech-lexicon.txt

.. _`LibriSpeech data set`: https://www.dropbox.com/s/i08yunn7yqnbv0h/LibriSpeech.zip?dl=0

********
Examples
********

Example 1: LibriSpeech
======================

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

Example 2: A complex command
============================

From the root directory of the Montreal Forced Aligner:

.. code-block:: bash

   bin/mfa_train_and_align ~/2_French_training ~/French/fr-QuEu.dict ~/2_French_aligned -s 7 -f -v

This command will train a new model and align the files in ``~/2_French_training``
using the dictionary file ``~/French/fr-QuEu.dict``, and save the output
TextGrids to ``~/2_French_training``.  It will take the first 7 characters
of the file name to be the speaker ID number.  It will be fast (do half
as many training iterations) and verbose (output more info to Terminal during training).
