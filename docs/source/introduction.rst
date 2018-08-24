
.. _`Kaldi homepage`: http://kaldi-asr.org/

.. _`Kaldi feature and model-space transforms page`: http://kaldi-asr.org/doc/transform.html

.. _`Phonetisaurus repository`: https://github.com/AdolfVonKleist/Phonetisaurus

.. _`HTK homepage`: http://htk.eng.cam.ac.uk/

.. _`Prosodylab-aligner homepage`: http://prosodylab.org/tools/aligner/

.. _`P2FA homepage`: https://www.ling.upenn.edu/phonetics/old_website_2015/p2fa/

.. _`FAVE-align homepage`: https://github.com/JoFrhwld/FAVE/wiki/FAVE-align

.. _`MAUS homepage`: http://www.bas.uni-muenchen.de/Bas/BasMAUS.html

.. _`Praat homepage`: http://www.fon.hum.uva.nl/praat/

.. _`EasyAlign homepage`: http://latlcui.unige.ch/phonetique/easyalign.php

.. _`Gentle homepage`: https://lowerquality.com/gentle/

.. _`@wavable`: https://twitter.com/wavable

.. _`Github`: http://mmcauliffe.github.io/

.. _`mailing list`: https://groups.google.com/forum/#!forum/mfa-users

.. _introduction:

************
Introduction
************

What is forced alignment?
=========================

Forced alignment is a technique to take an orthographic transcription of
an audio file and generate a time-aligned version using a pronunciation
dictionary to look up phones for words.

Many languages have :ref:`pretrained_acoustic` available for download and use.


Montreal Forced Aligner
=======================

Pipeline of training
--------------------

The Montreal Forced Aligner by default goes through four primary stages of training.  The
first pass of alignment uses monophone models, where each phone is modelled
the same regardless of phonological context.  The second pass uses triphone
models, where context on either side of a phone is taken into account for
acoustic models. The third pass performs LDA+MLLT to learn a transform of the features
that makes each phone's features maximally different. The final pass enhances the triphone model by taking
into account speaker differences, and calculates a transformation of the
mel frequency cepstrum coefficients (MFCC) features for each speaker.  See the `Kaldi feature and model-space transforms page`_
for more detail on these final passes.
The Montreal Forced Aligner can also train using deep neural networks (DNNs).

For more technical information about the structure of the aligner, see :ref:`alignment_techniques` and
:ref:`api_reference`.

If you run into any issues, please check the `mailing list`_ for fixes/workarounds or to post a new issue.

Use of speaker information
--------------------------

A key feature of the Montreal Forced Aligner is the use of speaker
adaptatation in alignment.  The command line interface provides multiple
ways of grouping audio files by speaker, depending on the input file format
(either :ref:`prosodylab_format` or :ref:`textgrid_format`).
In addition to speaker-adaptation in the final pass of alignment, speaker
information is used for grouping audio files together for multiprocessing
and ceptstral mean and variance normalization (CMVN).  If speakers are not
properly specified, then feature calculation might not succeed due to
limits on the numbers of files open.

Underlying technology
---------------------

The Montreal Forced Aligner uses the Kaldi ASR toolkit
(`Kaldi homepage`_) to perform forced alignment.
Kaldi is under active development and uses modern ASR and includes state-of-the-art algorithms for tasks
in automatic speech recognition beyond forced alignment.  For grapheme-to-phoneme capabilities, MFA uses Phonetisaurus
(`Phonetisaurus repository`_).

Other forced alignment tools
============================

Most tools for forced alignment used by linguists rely on the HMM Toolkit
(HTK; `HTK homepage`_), including:

* Prosodylab-aligner (`Prosodylab-aligner homepage`_)
* Penn Phonetics Forced Aligner (P2FA, `P2FA homepage`_)
* FAVE-align (`FAVE-align homepage`_)
* (Web) MAUS (`MAUS homepage`_)

EasyAlign (`EasyAlign homepage`_) is a Praat (`Praat homepage`_) plug-in for forced alignment as well.

Montreal Forced Aligner is most similar to the Prosodylab-aligner, and
was developed at the same lab.  Because the Montreal Forced Aligner uses
a different toolkit to do alignment, trained models cannot be used with
the Prosodylab-aligner, and vice versa.

Another Kaldi-based forced aligner is Gentle (`Gentle homepage`_) which uses Kaldi's neural networks to
align English data.  The Montreal Forced Aligner allows for training on any data that you might have, and
can be used with languages other than English.

Contributors
============

* Michael McAuliffe (michael.e.mcauliffe@gmail.com, `Github`_, `@wavable`_)
* Michaela Socolof
* Elias Stengel-Eskin
* Sarah Mihuc
* Arlie Coles
* Michael Wagner
* Morgan Sonderegger

Citation
========

McAuliffe, Michael, Michaela Socolof, Sarah Mihuc, Michael Wagner, and Morgan Sonderegger (2017).
Montreal Forced Aligner [Computer program]. Version 0.9.0,
retrieved 17 January 2017 from http://montrealcorpustools.github.io/Montreal-Forced-Aligner/.

Or:

McAuliffe, Michael, Michaela Socolof, Sarah Mihuc, Michael Wagner, and Morgan Sonderegger (2017).
Montreal Forced Aligner: trainable text-speech alignment using Kaldi. In
*Proceedings of the 18th Conference of the International Speech Communication Association*. :download:`Paper PDF <_static/MFA_paper_Interspeech2017.pdf>`


Funding
=======

We acknowledge funding from Social Sciences and Humanities Research Council (SSHRC) #430-2014-00018, Fonds de Recherche du Québec – Société et Culture (FRQSC) #183356 and Canada Foundation for Innovation (CFI) #32451 to Morgan Sonderegger.

