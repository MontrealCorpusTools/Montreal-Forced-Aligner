
.. _`LibriSpeech lexicon`: http://www.openslr.org/resources/11/librispeech-lexicon.txt

.. _`CMU Pronouncing Dictionary`: http://www.speech.cs.cmu.edu/cgi-bin/cmudict

.. _`Prosodylab-aligner English dictionary`: https://github.com/prosodylab/Prosodylab-Aligner/blob/master/eng.dict

.. _`Prosodylab-aligner French dictionary`: https://github.com/prosodylab/prosodylab-alignermodels/blob/master/FrenchQuEu/fr-QuEu.dict

.. _dictionary_format:

*****************
Dictionary format
*****************

.. _text_normalization:

Text normalization and dictionary lookup
========================================

If a word is not found in the dictionary, and has no orthographic
markers for morpheme boundaries (apostrophes or hyphens), then it will
be replaced in the output with '<unk>' for unknown word.

.. note::

   The list of all unknown words (out of vocabulary words; OOV words) will
   be output to a file named ``oovs_found.txt``
   in the output directory, if you would like to add them to the dictionary
   you are using.  To help find any typos in transcriptions, a file named
   ``utterance_oovs.txt`` will be put in the output directory and will list
   the unknown words per utterance.

As part of parsing orthographic transcriptions, punctuation is stripped
from the ends and beginnings of words, except for the :code:`brackets` specified in :ref:`configuration_dictionary`.  In addition, all words are converted to lowercase so that dictionary lookup is not case-sensitive.

.. note::

   The definition of punctuation, clitic markers, and compound markers can be set in a config file, see :ref:`configuration_dictionary` for more details.

Dictionary lookup will attempt to generate the most maximal coverage of
novel forms if they use some overt morpheme boundary in the orthography.

For instance, in French, clitics are marked using apostrophes between the
bound clitic and the stem.  Thus given a dictionary like:

.. highlight:: none

::

   c'est S E
   c S E
   c' S
   etait E T E
   un A N

And two example orthographic transcriptions:

::

   c'est un c
   c'etait un c

The normalization would result in the following:

::

   c'est un c
   c' était un c

With a pronunciation of:

::

   S E A N S E
   S E T E A N S E

The key point to note is that the pronunciation of the clitic ``c'`` is ``S``
and the pronunciation of the letter ``c`` in French is ``S A``.

The algorithm will try to associate the clitic marker with either the element
before (as for French clitics) or the element after (as for English clitics
like the possessive marker).  The default clitic markers are ``'`` and ``’`` (but they are collapsed into a single
clitic marker, ``'`` by default).

The default compound marker is a hyphen (``-``).
Compound markers are treated similarly to clitic markers, but they are not associated with one
particular element in the word over another.  Instead, they are used to simply split the compound word.
For example, ``merry-go-round`` will
become ``merry go round`` if the hyphenated form is not in the dictionary.
If no words are found on splitting the word based on hyphens or apostrophes,
then the word will be treated as a single unit (single unknown word).

The default behavior of the aligner to is to clean up these internal splits and reconstruct the original word.  If this is not desirable, you can disable clean up via the :code:`--disable_textgrid_cleanup` flag (see :ref:`configuration`).

Non-probabilistic format
========================

Dictionaries should be specified in the following format:

::

  WORDA PHONEA PHONEB
  WORDA PHONEC
  WORDB PHONEB PHONEC

where each line is a word with a transcription separated by white space.
Each phone in the transcription should be separated by white space as well.

A dictionary for English that has good coverage is the lexicon derived
from the LibriSpeech corpus (`LibriSpeech lexicon`_).
This lexicon uses the Arpabet transcription format (like the `CMU Pronouncing Dictionary`_).

The Prosodylab-aligner has two preconstructed dictionaries as well, one
for English (`Prosodylab-aligner English dictionary`_)
and one for Quebec French (`Prosodylab-aligner French dictionary`_), also see :ref:`pretrained_dictionaries` for a list of supported dictionaries.

.. note::

   See the page on :ref:`g2p_dictionary_generating` for how to use G2P models to generate a dictionary
   from our pretrained models or how to generate pronunciation dictionaries from orthographies.

Dictionaries with pronunciation probability
===========================================

Dictionaries can be parsed with pronunciation probabilities, usually as the output of :ref:`training_dictionary`.

The format for this dictionary format is:

::

  WORDA 1.0 PHONEA PHONEB
  WORDA 0.3 PHONEC
  WORDB 1.0 PHONEB PHONEC


.. note::

   The most likely probability for a word is set to 1.0 in the algorithm implemented in :ref:`training_dictionary`.
   While this means that the sum of probabilities per word is greater than 1, it does not penalize words for having
   multiple pronunciations, and these probabilities are converted to log costs in the eventual weighted FST.

Non-speech annotations
======================

There are two special phones that can be used for annotations that are not speech, ``sil`` and ``spn``.  The ``sil`` phone is used
to model silence, and the ``spn`` phone is used to model unknown words.  If you have annotations for non-speech vocalizations that are
similar to silence like breathing or exhalation, you can use the ``sil`` phone to align those.  You can use the ``spn`` phone
to align annotations like laughter, coughing, etc.

::

  {LG} spn
  {SL} sil


.. _speaker_dictionaries:

Per-speaker dictionaries
========================

In addition to specifying a single dictionary to use when aligning or transcribing, MFA also supports specifying per-speaker
dictionaries via a yaml file, like the following.

.. code-block:: yaml

   default: /mnt/d/Data/speech/english_us_ipa.txt

   speaker_a: /mnt/d/Data/speech/english_uk_ipa.txt
   speaker_b: /mnt/d/Data/speech/english_uk_ipa.txt
   speaker_c: /mnt/d/Data/speech/english_uk_ipa.txt

What the above yaml file specifies is a "default" dictionary that will be used for any speaker not explicitly listed with
another dictionary, so it's possible to train/align/transcribe using multiple dialects or languages, provided the model
specified is compatible with all dictionaries.

The way to use this per-speaker dictionary is in place of where the dictionary argument is:

.. code-block::

   mfa align /path/to/corpus /path/to/speaker_dictionaries.yaml /path/to/acoustic_model.zip /path/to/output


.. _multilingual_ipa:

Multilingual IPA mode
=====================

For the purposes of training multilingual models with IPA, there is a flag for training that enables this mode
:code:`--multilingual_ipa`. With this mode, it strips out certain diacritics that are not generally related to the quality
of the vowel (i.e., diacritics related to length), and splits digraphs (affricates and diphthongs/triphthongs) into
their component symbols.  The reasoning behind these are that length can be modelled through transition probabilities
and the model can take advantage of a less confusable phone set, and for digraphs, we can model the components parts
to account better for rarer sounds.  For example, in English /dʒ/ is more common than /ʒ/, so by modelling /dʒ/ as /d ʒ/,
we have more data for the solo instances of /ʒ/.  The downside for this split is that that the minimum duration is increased
to 6 frames (3 frames per phone), however in general, diphthongs and affricates should be longer than less complex sounds.

The default configuration for multilingual IPA is as follows:

.. code-block:: yaml

   strip_diacritics:
     - "ː"  # long, i.e. /ɑː/
     - "ˑ"  # half long, i.e. /ɑˑ/
     - "̆"  # extra short, i.e. /ĭ/
     - "̯"   # non syllabic, i.e. /i̯/
     - "͡"  # linking, i.e. /d͡ʒ/
     - "‿"  # linking, i.e. /d‿ʒ/
     - "͜"  # linking, i.e. /d͜ʒ/
     - "̩"   # syllabic, i.e. /n̩/

   digraphs:
     - "[dt][szʒʃʐʑʂɕç]" # affricates
     - "[aoɔe][ʊɪ]" # diphthongs

.. note::
   Digraphs are specified as a regular expression pattern, where the characters in first set of square brackets (i.e. ``[aoɔe]``)
   is the the set of characters that matches the first element in the digraph, and the characters in second set of square
   brackets (i.e. ``[ʊɪ]``) matches the second element.  Triphthongs or longer sequences can be specified with more
   sets of square brackets, like ``[e][i][u]`` would match just a /eiu/ triphthong.
