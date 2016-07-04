.. _tutorial:

.. _`LibriSpeech lexicon`: http://www.openslr.org/resources/11/librispeech-lexicon.txt

.. _`LibriSpeech corpus`: http://www.openslr.org/12/

.. _`CMU Pronouncing Dictionary`: http://www.speech.cs.cmu.edu/cgi-bin/cmudict

.. _`Prosodylab-aligner English dictionary`: https://github.com/prosodylab/Prosodylab-Aligner/blob/master/eng.dict

.. _`Prosodylab-aligner French dictionary`: https://github.com/prosodylab/prosodylab-alignermodels/blob/master/FrenchQuEu/fr-QuEu.dict

********
Tutorial
********

There are two modes for the Montreal Forced Aligner:

1. Use a pretrained model to align a data set (``mfa_align``)

2. Align a data set using only that data set (``mfa_train_and_align``) and
   optionally output the trained model for future use

The Montreal Forced Aligner supports two data formats:

1. Prosodylab-Aligner format (single channel sound files and corresponding orthographic
   transcriptions in .lab files with speaker designations specified)

2. Textgrid format (mono/stereo sound files and corresponding TextGrids where
   each speaker has a tier and each interval contains the orthographic
   transcription)

Dictionaries
============

Dictionaries should be specified in the following format:

::

  WORDA PHONEA PHONEB
  WORDB PHONEB PHONEC

Where each line is a word with a transcription separated by white space.
Each phone should be separated by white space as well.

A dictionary for English that has good coverage is the lexicon derived
from the LibriSpeech corpus (`LibriSpeech lexicon`_).
This lexicon uses the Arpabet transcription format (like the `CMU Pronouncing Dictionary`_).

There is an option when running the aligner for not using a dictionary (`--nodict`).
When run in this mode, the aligner will construct pronunciations for words
in the corpus based off their orthographies.  In this mode, a dataset with an example transcription

::

  WORDA WORDB

for a sound file would have the following dictionary generated:

::

  WORDA W O R D A
  WORDB W O R D B

The Prosodylab-aligner has two preconstructed dictionaries as well, one
for English (`Prosodylab-aligner English dictionary`_)
and one for Quebec French (`Prosodylab-aligner French dictionary`_)

Data formats
============

Prosodylab-Aligner format
-------------------------

Things you need before you can align:

1. Every .wav sound file you are aligning must have a corresponding .lab
   file which contains the text transcription of that .wav file.  The .wav and
   .lab files must have the same name. For example, if you have ``givrep_1027_2_1.wav``,
   its transcription should be in ``givrep_1027_2_1.lab`` (which is just a
   text file with the .lab extension). If you have transcriptions in a
   tab-separated text file (or an Excel file which can be saved as one),
   you can generate .lab files from it using the relabel function of relabel_clean.py.
   The relabel_clean.py script is currently in the prosodylab.alignertools repository on GitHub.

2. These .lab files do not have be in the same case as the words in the dictionary
   (i.e. all words are coerced to lower case), and punctuation is ignored.

3. You also need a pronunciation dictionary for the language you're
   aligning.  Our dictionaries for English and French are provided with
   the old Prosodylab Aligner (French is in prosodylab.alignermodels).
   You can also write your own dictionary or download others.


TextGrid format
---------------



Running the aligner
===================

Align using pretrained models
-----------------------------

The Montreal Forced Aligner comes with pretrained models/dictionaries for:

- English - trained from the LibriSpeech data set (`LibriSpeech corpus`_)
- Quebec French

Steps to align:



Align using only the data set
-----------------------------

Steps to align:

1. Open terminal, and change directory to montreal-forced-aligner.

2. Type ``bin/mfa_train_and_align`` followed by the arguments described
   above in Usage.  (On Mac/Unix, to save time typing out the path, you
   can drag a folder from Finder into Terminal and it will put the full
   path to that folder into your command.)


A template command:

.. code-block:: bash

   bin/mfa_train_and_align -s [#] [corpus-folder] [dictionary] [output-folder]

This command will train a new model and align the files in [corpus-folder]
using the file [dictionary], and save the output TextGrids to [output-folder].
It will take the first [#] characters of the file name to be the speaker ID number.

An example command:

.. code-block:: bash

   bin/mfa_train_and_align -s 7 ~/2_French_training ~/French/fr-QuEu.dict ~/2_French_training -f -v

This command will train a new model and align the files in ``~/2_French_training``
using the dictionary file ``~/French/fr-QuEu.dict``, and save the output
TextGrids to ``~/2_French_training``.  It will take the first 7 characters
of the file name to be the speaker ID number.  It will be fast (do half
as many training iterations) and verbose (output more info to Terminal during training).

3. Once the aligner finishes, the resulting TextGrids will be in the
   specified output directory.  Training can take a couple hours for large datasets.
