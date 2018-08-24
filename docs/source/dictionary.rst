
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

The Prosodylab-aligner has two preconstructed dictionaries as well, one
for English (`Prosodylab-aligner English dictionary`_)
and one for Quebec French (`Prosodylab-aligner French dictionary`_), also see `dictionaries`_ for a list of supported dictionaries.

.. note::

   See the page on :doc:`generating dictionaries <dictionary_generating>` for how to use G2P models to generate a dictionary
   from our pretrained models or how to generate pronunciation dictionaries from orthographies.

Non-speech annotations
----------------------

There are two special phones that can be used for annotations that are not speech, ``sil`` and ``spn``.  The ``sil`` phone is used
to model silence, and the ``spn`` phone is used to model unknown words.  If you have annotations for non-speech vocalizations that are
similar to silence like breathing or exhalation, you can use the ``sil`` phone to align those.  You can use the ``spn`` phone
to align annotations like laughter, coughing, etc.

::

  {LG} spn
  {SL} sil
