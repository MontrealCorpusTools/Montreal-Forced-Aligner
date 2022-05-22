.. _training_dictionary:

Add probabilities to a dictionary ``(mfa train_dictionary)``
============================================================

MFA includes a utility command for training :term:`pronunciation probabilities` of a dictionary given a corpus for alignment.

The implementation used here follow Kaldi's :kaldi_steps:`get_prons`, :kaldi_utils:`dict_dir_add_pronprobs.sh`, and :kaldi_utils:`lang/make_lexicon_fst_silprob.py`.

.. seealso::

   For a more in depth description of the algorithm, see the `Chen et al (2015) <https://www.danielpovey.com/files/2015_interspeech_silprob.pdf>`_

Example
-------

As an example, consider the following English and Japanese sentences:

.. tab-set::

   .. tab-item:: English
      :sync: english

      The red fox has read many books, but there's always more to read.

      Normalized:

      the red fox has read many books but there 's always more to read

   .. tab-item:: Japanese
      :sync: japanese

      アカギツネはいろんな本を読んできましたけど、まだまだ読み切りがありません。

      Normalized:

      アカギツネ は いろんな 本 を 読んで き ました けれども まだまだ 読み 切り が あり ません

The following pronunciation dictionaries:

.. tab-set::

   .. tab-item:: English
      :sync: english

      .. csv-table:: English US pronunciation dictionary
         :widths: 30, 70
         :header: "Word","Pronunciation"

         In addition to lexical variants for the present and past tense of "read", function words have several variants listed. The genitive marker "'s" has variants to account for stem-final voicing (:ipa_inline:`[s]` and :ipa_inline:`[z]`) and stem-final alveolar obstruents (:ipa_inline:`[ɪ z]`). The negative conjuction "but" has variants for the pronunciation of the vowel and final :ipa_inline:`/t/` as :ipa_inline:`[ʔ]` or :ipa_inline:`[ɾ]`. Likewise, the preposition "to" has variants for the initial :ipa_inline:`/t/` and vowel reductions.  The definite determiner "the" and distal demonstrative "there" have variants for stopping :ipa_inline:`/ð/` to :ipa_inline:`[d̪]`, along with reductions for vowels.

         "'s","s"
         "'s","z"
         "'s","ɪ z"
         "always","ɒː ɫ w ej z"
         "always","ɑː ɫ w ej z"
         "always","ɒː w ej z"
         "always","ɑː w ej z"
         "books","b ʊ k s"
         "but","b ɐ t"
         "but","b ɐ ʔ"
         "but","b ə ɾ"
         "fox","f ɑː k s"
         "has","h æ s"
         "has","h æ z"
         "many","m ɛ ɲ i"
         "more","m ɒː ɹ"
         "read","ɹ iː d"
         "read","ɹ ɛ d"
         "red","ɹ ɛ d"
         "the","d̪ iː"
         "the","d̪ iː ʔ"
         "the","d̪ ə"
         "the","ð iː"
         "the","ð iː ʔ"
         "the","ð ə"
         "there","d̪ ɚ"
         "there","d̪ ɛ ɹ"
         "there","ð ɚ"
         "there","ð ɛ ɹ"
         "to","t ə"
         "to","tʰ ʉː"
         "to","tʰ ʊ"
         "to","ɾ ə"


   .. tab-item:: Japanese
      :sync: japanese

      The main pronunciation variants are in the topic particle "は", the object particle "を", past tense polite suffix "ました", and the "but" conjunction "けれども". The particles are always pronounced as :ipa_inline:`[w a]` and :ipa_inline:`[o]` and never as their hiragana readings :ipa_inline:`[h a]` and :ipa_inline:`[w o]`, respectively.  For "ました", I've included various levels of devoicing for :ipa_inline:`/i/` between the voiceless obstruents from full voiced :ipa_inline:`[i]`, to devoiced :ipa_inline:`[i̥]` to deleted.

      .. csv-table:: Japanese pronunciation dictionary
         :widths: 30, 70
         :header: "Word","Pronunciation"

         "アカギツネ","a k a ɟ i ts ɨ n e"
         "は","h a"
         "は","w a"
         "いろんな","i ɾ o nː a"
         "本","h o ɴ"
         "を","o"
         "を","w o"
         "読んで","j o n d e"
         "き","c i"
         "ました","m a ɕ i̥ t a"
         "ました","m a ɕ i t a"
         "ました","m a ɕ t a"
         "けれども","k e ɾ e d o m o"
         "けれども","k e d o m o"
         "けれども","k e d o"
         "読み","j o m i"
         "切り","c i ɾ i"
         "が","ɡ a"
         "あり","a ɾ i"
         "ません","m a s e ɴ"

The basic steps to calculating pronunciation and silence probabilities is as follows:

1. Generate word-pronunciation pairs from the alignment lattices

The resulting dictionary can then be used as a dictionary for alignment or transcription.


Command reference
-----------------

.. autoprogram:: montreal_forced_aligner.command_line.mfa:create_parser()
   :prog: mfa
   :start_command: train_dictionary
