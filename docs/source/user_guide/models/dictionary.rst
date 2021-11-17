
.. _`English pronunciation dictionary`:  https://raw.githubusercontent.com/MontrealCorpusTools/mfa-models/main/dictionary/english.dict
.. _`French Prosodylab dictionary`:  https://raw.githubusercontent.com/MontrealCorpusTools/mfa-models/main/dictionary/fr.dict
.. _`German Prosodylab dictionary`:  https://raw.githubusercontent.com/MontrealCorpusTools/mfa-models/main/dictionary/de.dict
.. _`TalnUPF Spanish IPA dictionary`:  https://raw.githubusercontent.com/TalnUPF/phonetic_lexica/master/es/es_lexicon-IPA.txt
.. _`TalnUPF Spanish gpA dictionary`:  https://raw.githubusercontent.com/TalnUPF/phonetic_lexica/master/es/es_lexicon-gpA.txt
.. _`TalnUPF Catalan IPA dictionary`:  https://raw.githubusercontent.com/TalnUPF/phonetic_lexica/master/ca/ca_lexicon-IPA.txt
.. _`FalaBrasil dictionary`: https://gitlab.com/fb-nlp/nlp-resources/-/tree/main/res

.. _pretrained_dictionaries:

************************************
Available pronunciation dictionaries
************************************

Any of the following pronunciation dictionaries can be downloaded with the command :code:`mfa model download dictionary <language_id>`.  You
can get a full list of the currently available dictionaries via :code:`mfa model download dictionary`.  New dictionaries contributed
by users will be periodically added. If you would like to contribute your dictionaries, please contact Michael McAuliffe
at michael.e.mcauliffe@gmail.com.

.. csv-table::
   :header: "Language", "Link", "Orthography system", "Phone set"

   "English", `English pronunciation dictionary`_ , "Latin", "Arpabet (stressed)"
   "French", `French Prosodylab dictionary`_, "Latin", "Prosodylab French"
   "German", `German Prosodylab dictionary`_, "Latin", "Prosodylab German"
   "Brazilian Portuguese", `FalaBrasil dictionary`_, "Latin", ""
   "Spanish", `TalnUPF Spanish IPA dictionary`_, "Latin", "IPA"
   "Spanish", `TalnUPF Spanish gpA dictionary`_, "Latin", "gpA"
   "Catalan", `TalnUPF Catalan IPA dictionary`_, "Latin", "IPA"
