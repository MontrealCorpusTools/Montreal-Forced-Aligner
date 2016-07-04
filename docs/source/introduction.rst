.. _introduction:

.. _`Kaldi homepage`: http://kaldi-asr.org/

.. _`HTK homepage`: http://htk.eng.cam.ac.uk/

.. _`Prosodylab-aligner homepage`: http://prosodylab.org/tools/aligner/

.. _`P2FA homepage`: https://www.ling.upenn.edu/phonetics/old_website_2015/p2fa/

.. _`FAVE-align homepage`: http://fave.ling.upenn.edu/FAAValign.html

.. _`MAUS homepage`: http://www.bas.uni-muenchen.de/Bas/BasMAUS.html

.. _`Praat homepage`: http://www.fon.hum.uva.nl/praat/

.. _`EasyAlign homepage`: http://latlcui.unige.ch/phonetique/easyalign.php

Introduction
============

What is forced alignment?
-------------------------

Forced alignment is a technique to take an orthographic transcription of
an audio file and generate a time-aligned version using a pronunciation
dictionary to look up phones for words.

Underlying technology
---------------------

The Montreal Forced Aligner uses the Kaldi ASR toolkit
(`Kaldi homepage`_) to perform forced alignment.
Kaldi is under active development and uses modern ASR and includes state-of-the-art algorithms for tasks
in automatic speech recognition beyond forced alignment.

Relation to other forced alignment tools
----------------------------------------

Most tools for forced alignment used by linguists rely on the HMM Toolkit
(HTK; `HTK homepage`_), including:

* Prosodylab-aligner (`Prosodylab-aligner homepage`_)
* Penn Phonetics Forced Aligner (P2FA, `P2FA homepage`_)
* FAVE-align (`FAVE-align homepage`_)
* (Web) MAUS(`MAUS homepage`_)

Praat (`Praat homepage`_)
has a built-in aligner as well.
EasyAlign (`EasyAlign homepage`_)
is a Praat plug-in built to facilitate its use.




Contributors
------------

* Michael McAuliffe
* Michaela Socolof
* Sarah Mihuc
* Michael Wagner

Citation
--------

McAuliffe, Michael, Michaela Socolof, Sarah Mihuc, and Michael Wagner (2016).
Montreal Forced Aligner [Computer program]. Version 0.5,
retrieved 13 July 2016 from http://montrealcorpustools.github.io/Montreal-Forced-Aligner/.

Funding
-------

