

.. _`Pynini`: https://github.com/kylebgormon/Pynini
.. _`Sigmorphon 2020 G2P task baseline`: https://github.com/sigmorphon/2020/tree/master/task1/baselines/fst

.. _`ProsodyLab dictionary repository`: https://github.com/prosodylab/prosodylab.dictionaries

.. _`Lexique`: http://www.lexique.org/

.. _`ProsodyLab French dictionary`: https://github.com/prosodylab/prosodylab.dictionaries/raw/master/fr.dict

.. _pretrained_g2p:

*********************
Pretrained G2P models
*********************


Included with MFA is a separate tool to generate a dictionary from a preexisting model. This should be used if you're
aligning a dataset for which you have no pronunciation dictionary or the orthography is very transparent. We have pretrained
models for several languages below.

Any of the following G2P models can be downloaded with the command :code:`mfa model download g2p <language_id>`.  You can get a full list of the currently available G2P models via :code:`mfa download g2p`.  New models contributed by users will be periodically added. If you would like to contribute your trained models, please contact Michael McAuliffe at michael.e.mcauliffe@gmail.com.

These models were generated using the `Pynini`_ package on the GlobalPhone dataset. The implementation is based on that in the
`Sigmorphon 2020 G2P task baseline`_.
This means that they will only work for transcriptions which use the same
alphabet. Current language options are listed below, with the following accuracies when trained on 90% of the data and
tested on 10%:

.. csv-table::
   :header: "Language", "Link", "WER", "LER", "Orthography system", "Phone set"

   "Arabic", "Use not recommended due to issues in GlobalPhone", 28.45, 7.42, "Romanized [2]_", "GlobalPhone"
   "Bulgarian", :mfa_model:`g2p/bulgarian_g2p`, 3.08, 0.38, "Cyrillic alphabet", "GlobalPhone"
   "Croatian", :mfa_model:`g2p/croatian_g2p`, 9.47, 3.4, "Latin alphabet", "GlobalPhone"
   "Czech", :mfa_model:`g2p/czech_g2p`, 3.43, 0.71, "Latin alphabet", "GlobalPhone"
   "English", :mfa_model:`g2p/english_g2p`, 28.45, 7.42, "Latin alphabet", "Arpabet"
   "French", :mfa_model:`g2p/french_g2p`, 42.54, 6.98, "Latin alphabet", "GlobalPhone"
   "French", :mfa_model:`g2p/french_lexique_g2p`, 5.31, 1.06, "Latin alphabet", "Lexique"
   "French", :mfa_model:`g2p/french_prosodylab_g2p` [1]_, 5.11, 0.95, "Latin alphabet", "Prosodylab"
   "German", :mfa_model:`g2p/german_g2p`, 36.16, 7.84, "Latin alphabet", "GlobalPhone"
   "German", :mfa_model:`g2p/german_prosodylab_g2p` [3]_, 5.43, 0.65, "Latin alphabet", "Prosodylab"
   "Hausa", :mfa_model:`g2p/hausa_g2p`, 32.54, 7.19, "Latin alphabet", "GlobalPhone"
   "Japanese", :mfa_model:`g2p/japanese_character_g2p`, 17.45, 7.17, "Kanji and kana", "GlobalPhone"
   "Korean", :mfa_model:`g2p/korean_hangul_g2p`, 11.85, 1.38, "Hangul", "GlobalPhone"
   "Korean", :mfa_model:`g2p/korean_jamo_g2p`, 8.94, 0.95, "Jamo", "GlobalPhone"
   "Mandarin", :mfa_model:`g2p/mandarin_pinyin_g2p`, 0.27, 0.06, "Pinyin", "Pinyin phones"
   "Mandarin", :mfa_model:`g2p/mandarin_character_g2p` [4]_, 23.81, 11.2, "Hanzi", "Pinyin phones [6]_"
   "Polish", :mfa_model:`g2p/polish_g2p`, 1.23, 0.33, "Latin alphabet", "GlobalPhone"
   "Portuguese", :mfa_model:`g2p/portuguese_g2p`, 10.67, 1.62, "Latin alphabet", "GlobalPhone"
   "Russian", :mfa_model:`g2p/russian_g2p`, 4.04, 0.65, "Cyrillic alphabet", "GlobalPhone"
   "Spanish", :mfa_model:`g2p/spanish_g2p`, 17.93, 3.02, "Latin alphabet", "GlobalPhone"
   "Swahili", :mfa_model:`g2p/swahili_g2p`, 0.09, 0.02, "Latin alphabet", "GlobalPhone"
   "Swedish", :mfa_model:`g2p/swedish_g2p`, 18.75, 3.14, "Latin alphabet", "GlobalPhone"
   "Thai", :mfa_model:`g2p/thai_g2p`, 27.62, 7.48, "Thai script", "GlobalPhone"
   "Turkish", :mfa_model:`g2p/turkish_g2p`, 8.51, 2.32, "Latin alphabet", "GlobalPhone"
   "Ukrainian", :mfa_model:`g2p/ukrainian_g2p`, 2.1, 0.42, "Cyrillic alphabet", "GlobalPhone"
   "Vietnamese", :mfa_model:`g2p/vietnamese_g2p`, 14.91, 3.46, "Vietnamese alphabet", "GlobalPhone"
   "Wu", :mfa_model:`g2p/wu_g2p` [5]_ , 31.19, 13.04, "Hanzi", "GlobalPhone"


.. [1] The `ProsodyLab French dictionary`_ is based on `Lexique`_ with substitutions for numbers and special characters.
   Note that Lexique is known to currently not work with the aligner, see the `Github issue <https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/issues/29>`_
   for more information and status.
.. [2] Please see the GlobalPhone documentation for how the romanization was done for Arabic.
.. [3] The German dictionary used in training is available in the `ProsodyLab dictionary repository`_.
   See http://www.let.uu.nl/~Hugo.Quene/personal/phonchar.html for more information on the CELEX phone set for German
   and how it maps to other phonesets.
.. [4] The Mandarin character dictionary that served as the training data for this model was built by mapping between
   characters in ``.trl`` files and pinyin syllables in ``.rmn`` files in the GlobalPhone corpus.
.. [5] The Wu G2P model was trained a fairly small lexicon, so it likely does not have the coverage to be a robust model
   for most purposes.  Please check carefully any resulting dictionaries, as they are likely to have missing syllables from
   from unknown symbols.
.. [6] The phoneset for Mandarin was created by GlobalPhone by splitting Pinyin into onset, nucleus (any vowel sequence),
   and codas, and then associating the tone of the syllable onto the nucleus (i.e. "fang2" -> "f a2 ng" and "xiao4" ->
   "x iao4"
