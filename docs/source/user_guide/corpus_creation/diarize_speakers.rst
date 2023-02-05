.. _diarize_speakers:

Speaker diarization ``(mfa diarize_speakers)``
==============================================

The Montreal Forced Aligner can use trained ivector models (see :ref:`train_ivector` for more information about training these models) to classify or cluster utterances according to speakers.

Following ivector extraction, MFA stores utterance and speaker ivectors in PLDA-transformed space.  Storing the PLDA transformation ensures that the transformation is performed only once when ivectors are initially extracted, rather than done each time scoring occurs. The dimensionality of the PLDA-transformed ivectors is 50, by default, but this can be changed through the :ref:`configure_cli` command.

.. seealso::

   The PLDA transformation and scoring generally follows `Probabilistic Linear Discriminant Analysis (PLDA) Explained by Prachi Singh <https://towardsdatascience.com/probabilistic-linear-discriminant-analysis-plda-explained-253b5effb96>`_ and `the associated code <https://github.com/prachiisc/PLDA_scoring>`_.

A number of clustering algorithms from `scikit-learn <https://scikit-learn.org/stable/modules/clustering.html>`_ are available to use as input, along with the default `hdbscan <https://hdbscan.readthedocs.io/en/latest/index.html>`_.  Specifying ``--use_plda`` will use PLDA scoring, as opposed to Euclidean distance in PLDA-transformed space.  PLDA scoring is likely better, but does have the drawback of computing the full distance matrix for ``hdbscan``, ``affinity``, ``agglomerative``, ``spectral``, ``dbscan``, and ``optics``.

.. warning::

   Some experimentation in clustering is likely necessary, and in general, should be run in a very supervised manner.  Different recording conditions and noise in particular utterances can affect the ivectors.  Please see the speaker diarization functionality of :xref:`anchor` for a way to run MFA's diarization in a supervised manner.

   Also, do note that much of the speaker diarization functionality in MFA is implemented particularly for Anchor, as it's not quite as constrained a problem as forced alignment.  As such, please consider speaker diarization from the command line as alpha functionality, there are likely to be issues.

Command reference
-----------------

.. click:: montreal_forced_aligner.command_line.diarize_speakers:diarize_speakers_cli
   :prog: mfa diarize_speakers
   :nested: full

Configuration reference
-----------------------

- :ref:`configuration_diarization`

API reference
-------------

- :ref:`diarization_api`
