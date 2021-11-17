

.. _user_guide:

**********
User Guide
**********

What is forced alignment?
=========================

Forced alignment is a technique to take an orthographic transcription of
an audio file and generate a time-aligned version using a pronunciation
dictionary to look up phones for words.

Many languages have :ref:`pretrained_acoustic_models` available for download and use.

.. note::

   For a more detailed background on forced alignment, please see Eleanor Chodroff's excellent :xref:`chodroff_kaldi` within her larger :xref:`chodroff_phonetics`.


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
mel frequency cepstrum coefficients (MFCC) features for each speaker.  See the :xref:`kaldi` page on feature transformations
for more detail on these final passes.

For more technical information about the structure of the aligner, see
:ref:`mfa_api`.

If you run into any issues, please check the :xref:`mfa_mailing_list` for fixes/workarounds or to post a new issue on in the :xref:`mfa_github_issues`.

Use of speaker information
--------------------------

A key feature of the Montreal Forced Aligner is the use of speaker
adaptation in alignment.  The command line interface provides multiple
ways of grouping audio files by speaker, depending on the input file format
(either :ref:`prosodylab_format` or :ref:`textgrid_format`).
In addition to speaker-adaptation in the final pass of alignment, speaker
information is used for grouping audio files together for multiprocessing
and cepstral mean and variance normalization (CMVN).  If speakers are not
properly specified, then feature calculation might not succeed due to
limits on the numbers of files open.

Underlying technology
---------------------

The Montreal Forced Aligner uses the :xref:`kaldi` ASR toolkit to perform forced alignment.
Kaldi is under active development and uses modern ASR and includes state-of-the-art algorithms for tasks
in automatic speech recognition beyond forced alignment.  For grapheme-to-phoneme capabilities, MFA 1.0 used :xref:`phonetisaurus`, but MFA 2.0 has switched to using :xref:`pynini`.

Other forced alignment tools
============================

Most tools for forced alignment used by linguists rely on the HMM Toolkit
(:xref:`htk`), including:

* :xref:`prosodylab_aligner`
* :xref:`p2fa`
* :xref:`fave`
* :xref:`maus`

:xref:`easy_align` is a :xref:`praat` plug-in for forced alignment as well.

Montreal Forced Aligner is most similar to the Prosodylab-aligner, and
was developed at the same lab.  Because the Montreal Forced Aligner uses
a different toolkit to do alignment, trained models cannot be used with
the Prosodylab-aligner, and vice versa.

Another Kaldi-based forced aligner is :xref:`gentle` which uses Kaldi's neural networks to
align English data.  The Montreal Forced Aligner allows for training on any data that you might have, and
can be used with languages other than English.

Contributors
============

* Michael McAuliffe

  - :fa:`envelope` michael.e.mcauliffe@gmail.com
  - :fa:`blog` :xref:`memcauliffe.com`
  - :fa:`twitter` :xref:`@wavable`

* :xref:`socolof`
* :xref:`stengel-eskin`
* :xref:`mihuc`
* :xref:`coles`
* :xref:`wagner`
* :xref:`sonderegger`

Citation
========

McAuliffe, Michael, Michaela Socolof, Sarah Mihuc, Michael Wagner, and Morgan Sonderegger (2017).
Montreal Forced Aligner [Computer program]. Version 0.9.0,
retrieved 17 January 2017 from http://montrealcorpustools.github.io/Montreal-Forced-Aligner/.

Or:

McAuliffe, Michael, Michaela Socolof, Sarah Mihuc, Michael Wagner, and Morgan Sonderegger (2017).
Montreal Forced Aligner: trainable text-speech alignment using Kaldi. In
*Proceedings of the 18th Conference of the International Speech Communication Association*. :download:`Paper PDF <../_static/MFA_paper_Interspeech2017.pdf>`


Funding
=======

We acknowledge funding from Social Sciences and Humanities Research Council (SSHRC) #430-2014-00018, Fonds de Recherche du Québec – Société et Culture (FRQSC) #183356 and Canada Foundation for Innovation (CFI) #32451 to Morgan Sonderegger.

.. toctree::
   :hidden:

   commands
   formats/index
   data_validation
   workflows/index
   configuration/index
   models/index