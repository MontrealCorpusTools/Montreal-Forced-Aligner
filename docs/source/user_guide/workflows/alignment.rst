
.. _pretrained_alignment:

Align with an acoustic model ``(mfa align)``
============================================

This is the primary workflow of MFA, where you can use pretrained :term:`acoustic models` to align your dataset.  There are a number of :xref:`pretrained_acoustic_models` to use, but you can also adapt a pretrained model to your data (see :ref:`adapt_acoustic_model`) or train an acoustic model from scratch using your dataset (see :ref:`train_acoustic_model`).

.. _alignment_evaluation:

Evaluation mode
---------------

Alignments can be compared to a gold-standard reference set by specifying the ``--reference_directory`` below. MFA will load all TextGrids and parse them as if they were exported by MFA (i.e., phone and speaker tiers per speaker).  The phone intervals will be aligned using the :mod:`Bio.pairwise2` alignment algorithm. If the reference TextGrids use a different phone set, then a custom mapping yaml file can be specified via the ``--custom_mapping_path``.  As an example, the Buckeye reference alignments used in `Update on Montreal Forced Aligner performance <https://memcauliffe.com/update-on-montreal-forced-aligner-performance.html>`_ use its own ARPA-based phone set that removes stress integers, is lower case, and has syllabic sonorants.  To map alignments generated with the ``english`` model and dictionary that use standard ARPA, a yaml file like the following allows for a better alignment of reference phones to aligned phones.

.. code-block:: yaml

   N: [en, n]
   M: [em, m]
   L: [el, l]
   AA0: aa
   AE0: ae
   AH0: ah
   AO0: ao
   AW0: aw

Using the above file, both ``en`` and ``n`` phones in the Buckeye corpus will not be penalized when matched with ``N`` phones output by MFA.

In addition to any custom mapping, phone boundaries are used in the cost function for the :mod:`Bio.pairwise2` alignment algorithm as follows:

.. math::

   Overlap \: cost = -1 * \biggl(\lvert begin_{aligned} - begin_{ref} \rvert + \lvert end_{aligned} - end_{ref} \rvert + \begin{cases}
            0, & label_{1} = label_{2} \\
            2, & otherwise
            \end{cases}\biggr)

The two metrics calculated for each utterance are overlap score and phone error rate.  Overlap score is calculated similarly to the above cost function for each phone (excluding phones that are aligned to silence or were inserted/deleted) and averaged over the utterance:

.. math::

   Alignment \: score = \frac{Overlap \: cost}{2}

Phone error rate is calculated as:

.. math::

   Phone \: error \: rate = \frac{insertions + deletions + (2 * substitutions)} {length_{ref}}

.. _phone_models:

Phone model alignments
----------------------

With the ``--use_phone_model`` flag, an ngram language model for phones will be constructed and used to generate phone transcripts with alignments.  The phone language model uses bigrams and higher orders (up to 4), with no unigrams included to speed up transcription (and because the phonotactics of languages highly constrain the possible sequences of phones).  The phone language model is trained on phone transcriptions extracted from alignments and includes silence and OOV phones.

The phone transcription additionally uses speaker-adaptation transforms from the regular alignment as well to speed up transcription.  From the phone transcription lattices, we extract phone-level alignments along with confidence score using :kaldi_src:`lattice-to-ctm-conf`.

The alignments extracted from phone transcriptions are compared to the baseline alignments using the procedure outlined in :ref:`alignment_evaluation` above.

.. _fine_tune_alignments:

Fine-tuning alignments
----------------------

By default and standard in ASR, the frame step between feature frames is set to 10 ms, which limits the accuracy of MFA to a minimum of 0.01 seconds. When the ``--fine_tune`` flag is specified, the aligner does an extra fine-tuning step following alignment. The audio surrounding each interval's initial boundary is extracted with a frame step of 1 ms (0.001s) and is aligned using a simple phone dictionary combined with a transcript of the previous phone and the current phone.  Extracting the phone alignment gives the possibility of higher degrees of accuracy (down to 1ms).

.. warning::

   The actual accuracy bound is not clear as each frame uses the surrounding 25ms to generate features, so each frame necessary incorporates time-smeared acoustic information.

Command reference
-----------------

.. click:: montreal_forced_aligner.command_line.align:align_corpus_cli
   :prog: mfa align
   :nested: full

Configuration reference
-----------------------

- :ref:`configuration_global`

API reference
-------------

- :ref:`alignment_api`
