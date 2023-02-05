
.. _`Chen et al (2015)`: https://www.danielpovey.com/files/2015_interspeech_silprob.pdf
.. _`English US MFA dictionary`: https://mfa-models.readthedocs.io/en/latest/dictionary/English/English%20%28US%29%20MFA%20dictionary%20v2_0_0a.html#English%20(US)%20MFA%20dictionary%20v2_0_0a
.. _`Japanese MFA dictionary`: https://mfa-models.readthedocs.io/en/latest/dictionary/Japanese/Japanese%20MFA%20dictionary%20v2_0_0.html#Japanese%20MFA%20dictionary%20v2_0_0

.. _training_dictionary:

Add probabilities to a dictionary ``(mfa train_dictionary)``
============================================================

MFA includes a utility command for training :term:`pronunciation probabilities` of a dictionary given a corpus for alignment.

The implementation used here follow Kaldi's :kaldi_steps:`get_prons`, :kaldi_utils:`dict_dir_add_pronprobs.sh`, and :kaldi_utils:`lang/make_lexicon_fst_silprob.py`.

.. seealso::

   Refer to the :ref:`lexicon FST concept section <lexicon_fst>` for an introduction and overview of how MFA compiles pronunciation dictionaries to a :term:`WFST`. The algorithm and calculations below are based on `Chen et al (2015)`_.

Consider the following :term:`WFST` with two pronunciations of "because" from the trained `English US MFA dictionary`_.


    .. figure:: ../../_static/because.svg
        :align: center
        :alt: :term:`FST` for two pronunciations of "the" in the English US dictionary

In the above figure, there are are two final states, with 0 corresponding to a word preceded by ``non-silence`` and 1 corresponding to a word preceded by ``silence``.  The costs associated with each transition are negative log-probabilities, so that less likely paths cost more.  The state 0 refers to the beginning of speech, so the paths to the silence and non silence state are equal in this case. The cost for ending on silence is lower at -0.77 than ending on non-silence with a cost of 1.66, meaning that most utterances in the training data had trailing silence at the end of the recordings.

.. seealso::

   See :ref:`probabilistic_lexicons` for more information on probabilities in lexicons.


Command reference
-----------------

.. click:: montreal_forced_aligner.command_line.train_dictionary:train_dictionary_cli
   :prog: mfa train_dictionary
   :nested: full
