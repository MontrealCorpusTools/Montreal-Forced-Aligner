
.. _pretrained_alignment:

Align with an acoustic model ``(mfa align)``
============================================

This is the primary workflow of MFA, where you can use pretrained :term:`acoustic models` to align your dataset.  There are a number of :xref:`pretrained_acoustic_models` to use, but you can also adapt a pretrained model to your data (see :ref:`adapt_acoustic_model`) or train an acoustic model from scratch using your dataset (see :ref:`train_acoustic_model`).

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

   Overlap \: score = \frac{\sum\limits_{i=0}^{n-1} (\lvert begin_{aligned[i]} - begin_{ref[i]} \rvert + \lvert end_{aligned[i]} - end_{ref[i]} \rvert )}{n}

Phone error rate is calculated as:

.. math::

   Phone \: error \: rate = \frac{insertions + deletions + (2 * substitutions)} {length_{ref}}


Command reference
-----------------

.. autoprogram:: montreal_forced_aligner.command_line.mfa:create_parser()
   :prog: mfa
   :start_command: align

Configuration reference
-----------------------

- :ref:`configuration_global`

API reference
-------------

- :ref:`alignment_api`
