
.. _`LibriSpeech lexicon`: http://www.openslr.org/resources/11/librispeech-lexicon.txt

.. _`CMU Pronouncing Dictionary`: http://www.speech.cs.cmu.edu/cgi-bin/cmudict

.. _`Prosodylab-aligner English dictionary`: https://github.com/prosodylab/Prosodylab-Aligner/blob/master/eng.dict

.. _`Prosodylab-aligner French dictionary`: https://github.com/prosodylab/prosodylab-alignermodels/blob/master/FrenchQuEu/fr-QuEu.dict

.. _dictionary:

************
Dictionaries
************

Dictionaries should be specified in the following format:

::

  WORDA PHONEA PHONEB
  WORDB PHONEB PHONEC

where each line is a word with a transcription separated by white space.
Each phone in the transcription should be separated by white space as well.

A dictionary for English that has good coverage is the lexicon derived
from the LibriSpeech corpus (`LibriSpeech lexicon`_).
This lexicon uses the Arpabet transcription format (like the `CMU Pronouncing Dictionary`_).

There is an option when running the aligner for not using a dictionary (``--nodict``).
When run in this mode, the aligner will construct pronunciations for words
in the corpus based on their orthographies.  In this mode, a dataset with an example transcription

::

  WORDA WORDB

for a sound file would have the following dictionary generated:

::

  WORDA W O R D A
  WORDB W O R D B

The Prosodylab-aligner has two preconstructed dictionaries as well, one
for English (`Prosodylab-aligner English dictionary`_)
and one for Quebec French (`Prosodylab-aligner French dictionary`_)

