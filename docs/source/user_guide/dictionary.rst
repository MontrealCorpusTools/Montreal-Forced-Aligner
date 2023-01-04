
.. _`LibriSpeech lexicon`: http://www.openslr.org/resources/11/librispeech-lexicon.txt

.. _`CMU Pronouncing Dictionary`: http://www.speech.cs.cmu.edu/cgi-bin/cmudict

.. _`Prosodylab-aligner English dictionary`: https://github.com/prosodylab/Prosodylab-Aligner/blob/master/eng.dict

.. _`Prosodylab-aligner French dictionary`: https://github.com/prosodylab/prosodylab-alignermodels/blob/master/FrenchQuEu/fr-QuEu.dict

.. _dictionary_format:

*******************************
Pronunciation dictionary format
*******************************

.. warning::

   As of 2.0.5, dictionaries have a firmer format of requiring tab-delimited columns (words, pronunciations, etc), and space-delimited pronunciations to avoid confusions in automatically interpreting dictionary format for phonesets that include numbers like X-SAMPA.

   If your dictionary uses spaces as the delimiter between orthography and pronunciations, you can re-encode it with tabs in a text editor that has regex search and replace support. The regex pattern :code:`^(\S+)\s+` replaced with :code:`\1\t` or :code:`$1\t`, depending on the text editor in question, will replace the first whitespace in every line with a tab.

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
   c     S E
   c'    S
   etait E T E
   un    A N

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
and the pronunciation of the letter ``c`` in French is ``S E``.

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

Dictionaries should be specified as a two column tab separated file:

::

  WORDA  PHONEA PHONEB
  WORDA  PHONEC
  WORDB  PHONEB PHONEC

Each line has a word and a transcription separated by a tab. Each phone in the transcription should be separated by non-tab white space as well.

.. note::

   Up to 2.0.5, words and their transcriptions could be separated by any white space, not just tab. However, given the complexities of parsing :ref:`probabilistic lexicons <probabilistic_dictionaries>` with transcription systems like X-SAMPA that use numbers as symbols, we have decided to standardize the column delimiter as tab.

A dictionary for English that has good coverage is the lexicon derived
from the LibriSpeech corpus (`LibriSpeech lexicon`_).
This lexicon uses the Arpabet transcription format (like the `CMU Pronouncing Dictionary`_).

The Prosodylab-aligner has two preconstructed dictionaries as well, one
for English (`Prosodylab-aligner English dictionary`_)
and one for Quebec French (`Prosodylab-aligner French dictionary`_), also see :xref:`pretrained_dictionaries` for a list of supported dictionaries.

.. note::

   See the page on :ref:`g2p_dictionary_generating` for how to use G2P models to generate a dictionary
   from our pretrained models or how to generate pronunciation dictionaries from orthographies.

.. _probabilistic_dictionaries:

Dictionaries with pronunciation probability
===========================================

Dictionaries can be parsed with pronunciation probabilities, usually as the output of :ref:`training_dictionary`.

The format for this dictionary format is:

::

  WORDA  1.0   PHONEA PHONEB
  WORDA  0.3   PHONEC
  WORDB  1.0   PHONEB PHONEC

The three columns should be separated by tabs, with the first column corresponding the orthographic form, the second to the pronunciation probability between 0.01 and 1.0, and the final column the space-delimited pronunciation.

.. note::

   The most likely probability for a word is set to 1.0 in the algorithm implemented in :ref:`training_dictionary`.
   While this means that the sum of probabilities per word is greater than 1, it does not penalize words for having
   multiple pronunciations, and these probabilities are converted to log costs in the eventual weighted FST.

Silence probabilities
---------------------

As part of modeling pronunciation probabilities, probabilities of silence before and after a given pronunciation can be estimated as well. As an example, with pronunciations of the English word ``the``, we might have one that is a full version ``[ð i]`` and a more reduced version ``[ð ə]``.  While the the more reduced version will be the more likely variant overall, the full version will likely have a higher probabilities following or preceding silence.

The format for this dictionary format is:

::

  the    0.16	   0.08	   2.17	   1.13	   d i
  the    0.99	   0.04	   2.14	   1.15	   d ə
  the	   0.01	   0.14	   2.48	   1.18	   ð i
  the	   0.02	   0.12	   1.87	   1.23	   ð ə
  the	   0.11	   0.15	   2.99	   1.15	   ə

The first float column is the probability of the pronunciation, the next float is the probability of silence following the pronunciation, and the final two floats are correction terms for preceding silence and non-silence. Given that each entry in a dictionary is independent and there is no way to encode information about the preceding context, the correction terms are calculated as how much more common was silence or non-silence compared to what we would expect factoring out the likelihood of silence from the previous word. More details are found in :kaldi_steps:`get_prons` and the `related paper <https://www.danielpovey.com/files/2015_interspeech_silprob.pdf>`_.

.. note::

   You can include entries that only have pronunciations or pronunciation probabilities mixed with those with silence probabilities.  If an entry doesn't have a pronunciation probability, it will default to ``1.0`` (assumes equal weight between pronunciation variants as above).  If an entry does not have the three silence numbers, then the probability following silence will use the default (:ref:`defaults to 0.5 for non-pretrained models <configuration_dictionary>`, or :ref:`whatever probability was estimated during training <silence_probability>`), along with no correction for when the pronunciation follows silence or non-silence.

Non-speech annotations
======================

There are two special phones that can be used for annotations that are not speech, ``sil`` and ``spn``.  The ``sil`` phone is used
to model silence, and the ``spn`` phone is used to model unknown words.  If you have annotations for non-speech vocalizations that are
similar to silence like breathing or exhalation, you can use the ``sil`` phone to align those.  You can use the ``spn`` phone
to align annotations like laughter, coughing, etc.

::

  {LG}   spn
  {SL}   sil

.. _cutoff_modeling:

Modeling cutoffs and hesitations
================================

Often in spontaneous speech, speakers will produce truncated or cut-off words of the following word/words. To help model this specific case, using the flag :code:`--use_cutoff_model` will enable a mode where pronunciations are generated for cutoff words matching one of the following criteria:

1. The cutoff word matches the pattern of :code:`{start_bracket}(cutoff|hes)`, where :code:`{start_bracket}` is the set of all left side brackets defined in :code:`brackets` (:ref:`configuration_dictionary`). The following word must not be an OOV or non-speech word (silence, laughter, another cutoff, etc).
2. The cutoff word matches the pattern of :code:`{start_bracket}(cutoff|hes)[-_](word){end_bracket}`, where start and end brackets are defined in :code:`brackets` (:ref:`configuration_dictionary`).  The :code:`word` will be used in place of the following word above, but needs to be present in the dictionary, otherwise the target word for the cutoff will default back to the following word.

The generated pronunciations

.. _speaker_dictionaries:

Per-speaker dictionaries
========================

In addition to specifying a single dictionary to use when aligning or transcribing, MFA also supports specifying per-speaker
dictionaries via a yaml file, like the following.

.. code-block:: yaml

   default: /mnt/d/Data/speech/english_us_mfa.dict

   speaker_a: /mnt/d/Data/speech/english_uk_mfa.dict
   speaker_b: /mnt/d/Data/speech/english_uk_mfa.dict
   speaker_c: /mnt/d/Data/speech/english_uk_mfa.dict

What the above yaml file specifies is a "default" dictionary that will be used for any speaker not explicitly listed with
another dictionary, so it's possible to train/align/transcribe using multiple dialects or languages, provided the model
specified is compatible with all dictionaries.

The way to use this per-speaker dictionary is in place of where the dictionary argument is:

.. code-block::

   mfa align /path/to/corpus /path/to/speaker_dictionaries.yaml /path/to/acoustic_model.zip /path/to/output

.. _phone_sets:

Supported phone sets
====================

In addition to the basic capabilities, specifying a phone set can aid in creating acoustic models that are better suited to the particular phones, with better contextual questions dependent on the place and manner of articulation for triphone modeling.
