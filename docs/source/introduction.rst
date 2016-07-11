
.. _`Kaldi homepage`: http://kaldi-asr.org/

.. _`HTK homepage`: http://htk.eng.cam.ac.uk/

.. _`Prosodylab-aligner homepage`: http://prosodylab.org/tools/aligner/

.. _`P2FA homepage`: https://www.ling.upenn.edu/phonetics/old_website_2015/p2fa/

.. _`FAVE-align homepage`: http://fave.ling.upenn.edu/FAAValign.html

.. _`MAUS homepage`: http://www.bas.uni-muenchen.de/Bas/BasMAUS.html

.. _`Praat homepage`: http://www.fon.hum.uva.nl/praat/

.. _`EasyAlign homepage`: http://latlcui.unige.ch/phonetique/easyalign.php

.. _`@wavable`: https://twitter.com/wavable

.. _`Github`: http://mmcauliffe.github.io/

.. _introduction:

************
Introduction
************

What is forced alignment?
=========================

Forced alignment is a technique to take an orthographic transcription of
an audio file and generate a time-aligned version using a pronunciation
dictionary to look up phones for words.


Montreal Forced Aligner
=======================

Pipeline of training
--------------------

The Montreal Forced Aligner goes through three stages of training.  The
first pass of alignment uses monophone models, where each phone is modelled
the same regardless of phonological context.  The second pass uses triphone
models, where context on either side of a phone is taken into account for
acoustic models.  The final pass enhances the triphone model by taking
into account speaker differences, and calculates a transformation of the
mel frequency cepstrum coefficients (MFCC) features for each speaker.

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
in automatic speech recognition beyond forced alignment.

Other forced alignment tools
============================

Most tools for forced alignment used by linguists rely on the HMM Toolkit
(HTK; `HTK homepage`_), including:

* Prosodylab-aligner (`Prosodylab-aligner homepage`_)
* Penn Phonetics Forced Aligner (P2FA, `P2FA homepage`_)
* FAVE-align (`FAVE-align homepage`_)
* (Web) MAUS(`MAUS homepage`_)

Praat (`Praat homepage`_) has a built-in aligner as well.
EasyAlign (`EasyAlign homepage`_) is a Praat plug-in built to facilitate its use.

Montreal Forced Aligner is most similar to the Prosodylab-aligner, and
was developed at the same lab.  Because the Montreal Forced Aligner uses
a different toolkit to do alignment, trained models cannot be used with
the Prosodylab-aligner, and vice versa.

Contributors
============

* Michael McAuliffe (michael.e.mcauliffe@gmail.com, `Github`_, `@wavable`_)
* Michaela Socolof
* Sarah Mihuc
* Michael Wagner

Citation
========

McAuliffe, Michael, Michaela Socolof, Sarah Mihuc, and Michael Wagner (2016).
Montreal Forced Aligner [Computer program]. Version 0.5,
retrieved 13 July 2016 from http://montrealcorpustools.github.io/Montreal-Forced-Aligner/.

Funding
=======

