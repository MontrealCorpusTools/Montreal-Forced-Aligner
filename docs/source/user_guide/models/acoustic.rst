

.. _`ProsodyLab dictionary repository`: https://github.com/prosodylab/prosodylab.dictionaries

.. _`Lexique`: http://www.lexique.org/

.. _`ProsodyLab French dictionary`: https://github.com/prosodylab/prosodylab.dictionaries/raw/master/fr.dict

.. _pretrained_acoustic_models:

**************************
Pretrained acoustic models
**************************

As part of using the Montreal Forced Aligner in our own research, we have trained acoustic models for a number of languages.
If you would like to use them, please download them below.  Please note the dictionary that they were trained with to
see more information about the phone set.  When using these with a pronunciation dictionary, the phone sets must be
compatible.  If the orthography of the language is transparent, it is likely that we have a G2P model that can be used
to generate the necessary pronunciation dictionary.

Any of the following acoustic models can be downloaded with the command :code:`mfa model download acoustic <language_id>`.  You
can get a full list of the currently available acoustic models via :code:`mfa model download acoustic`.  New models contributed
by users will be periodically added. If you would like to contribute your trained models, please contact Michael McAuliffe
at michael.e.mcauliffe@gmail.com.

.. csv-table::
   :header: "Language", "Link", "Corpus", "Number of speakers", "Audio (hours)", "Phone set"

   "Arabic", "Use not recommended due to issues in GlobalPhone", "GlobalPhone", 80, 19.0, "GlobalPhone"
   "Bulgarian", :mfa_model:`acoustic/bulgarian`, "GlobalPhone", 79, 21.4, "GlobalPhone"
   "Croatian", :mfa_model:`acoustic/croatian`, "GlobalPhone", 94, 15.9, "GlobalPhone"
   "Czech", :mfa_model:`acoustic/czech`, "GlobalPhone", 102, 31.7, "GlobalPhone"
   "English", :mfa_model:`acoustic/english`, "LibriSpeech", 2484, 982.3, "Arpabet (stressed)"
   "French (FR)", :mfa_model:`acoustic/french`, "GlobalPhone", 100, 26.9, "GlobalPhone"
   "French (FR)", :mfa_model:`acoustic/french_prosodylab`, "GlobalPhone", 100, 26.9, "Prosodylab [1]_"
   "French (QC)", :mfa_model:`acoustic/french_qc`, "Lab speech", "N/A", "N/A", "Prosodylab [1]_"
   "German", :mfa_model:`acoustic/german`, "GlobalPhone", 77, 18, "GlobalPhone"
   "German", :mfa_model:`acoustic/german_prosodylab`, "GlobalPhone", 77, 18, "Prosodylab [2]_"
   "Hausa", :mfa_model:`acoustic/hausa`, "GlobalPhone", 103, 8.7, "GlobalPhone"
   "Japanese", "Not available yet", "GlobalPhone", 144, 34, "GlobalPhone"
   "Korean", :mfa_model:`acoustic/korean`, "GlobalPhone", 101, 20.8, "GlobalPhone"
   "Mandarin", :mfa_model:`acoustic/mandarin`, "GlobalPhone", 132, 31.2, "Pinyin phones [3]_"
   "Polish", :mfa_model:`acoustic/polish`, "GlobalPhone", 99, 24.6, "GlobalPhone"
   "Portuguese", :mfa_model:`acoustic/portuguese`, "GlobalPhone", 101, 26.3, "GlobalPhone"
   "Russian", :mfa_model:`acoustic/russian`, "GlobalPhone", 115, 26.5, "GlobalPhone"
   "Spanish", :mfa_model:`acoustic/spanish`, "GlobalPhone", 102, 22.1, "GlobalPhone"
   "Swahili", :mfa_model:`acoustic/swahili`, "GlobalPhone", 70, 11.1, "GlobalPhone"
   "Swedish", :mfa_model:`acoustic/swedish`, "GlobalPhone", 98, 21.7, "GlobalPhone"
   "Tamil", "Not available yet", "GlobalPhone", "N/A", "N/A", "GlobalPhone"
   "Thai", :mfa_model:`acoustic/thai`, "GlobalPhone", 98, 28.2, "GlobalPhone"
   "Turkish", :mfa_model:`acoustic/turkish`, "GlobalPhone", 100, 17.1, "GlobalPhone"
   "Ukrainian", :mfa_model:`acoustic/ukrainian`, "GlobalPhone", 119, 14.1, "GlobalPhone"
   "Vietnamese", :mfa_model:`acoustic/vietnamese`, "GlobalPhone", 129, 19.7, "GlobalPhone"
   "Wu", "Not available yet", "GlobalPhone", 41, 9.3, "GlobalPhone"

.. [1] The `ProsodyLab French dictionary`_ is based on `Lexique`_ with substitutions for numbers and special characters. Note that Lexique is known to currently not work with the aligner, see the `Github issue <https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/issues/29>`_ for more information and status.
.. [2] The German dictionary used in training is available in the `ProsodyLab dictionary repository`_.
   See http://www.let.uu.nl/~Hugo.Quene/personal/phonchar.html for more information on the CELEX phone set for German and how it maps to other phonesets.
.. [3] The phoneset for Mandarin was created by GlobalPhone by splitting Pinyin into onset, nucleus (any vowel sequence),
   and codas, and then associating the tone of the syllable onto the nucleus (i.e. "fang2" -> "f a2 ng" and "xiao4" -> "x iao4"
