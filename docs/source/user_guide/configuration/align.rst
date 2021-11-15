
.. _configuration_alignment:

***********************
Alignment Configuration
***********************

Global options
==============

These options are used for aligning the full dataset (and as part of training).  Increasing the values of them will
allow for more relaxed restrictions on alignment.  Relaxing these restrictions can be particularly helpful for certain
kinds of files that are quite different from the training dataset (i.e., single word production data from experiments,
or longer stretches of audio).


.. csv-table::
   :header: "Parameter", "Default value", "Notes"
   :escape: '

   "beam", 10, "Initial beam width to use for alignment"
   "retry_beam", 40, "Beam width to use if initial alignment fails"
   "transition_scale", 1.0, "Multiplier to scale transition costs"
   "acoustic_scale", 0.1, "Multiplier to scale acoustic costs"
   "self_loop_scale", 0.1, "Multiplier to scale self loop costs"
   "boost_silence", 1.0, "1.0 is the value that does not affect probabilities"
   "punctuation", "、。।，@<>'"'(),.:;¿?¡!\\&%#*~【】，…‥「」『』〝〟″⟨⟩♪・‹›«»～′$+=", "Characters to treat as punctuation and strip from around words"
   "clitic_markers", "'''’", "Characters to treat as clitic markers, will be collapsed to the first character in the string"
   "compound_markers", "\-", "Characters to treat as marker in compound words (i.e., doesn't need to be preserved like for clitics)"
   "multilingual_ipa", False, "Flag for enabling multilingual IPA mode, see :ref:`multilingual_ipa` for more details"
   "strip_diacritics", "/iː/ /iˑ/ /ĭ/ /i̯/  /t͡s/ /t‿s/ /t͜s/ /n̩/", "IPA diacritics to strip in multilingual IPA mode (phone symbols for proper display, when specifying them just have the diacritic)"
   "digraphs", "[dt][szʒʃʐʑʂɕç], [aoɔe][ʊɪ]", "Digraphs to split up in multilingual IPA mode"


.. _feature_config:

Feature Configuration
=====================

This section is only relevant for training, as the trained model will contain extractors and feature specification for
what it requires.

.. csv-table::
   :header: "Parameter", "Default value", "Notes"

   "type", "mfcc", "Currently only MFCCs are supported"
   "use_energy", "False", "Use energy in place of first MFCC"
   "frame_shift", 10, "In milliseconds, determines time resolution"
   "snip_edges", True, "Should provide better time resolution in alignment"
   "pitch", False, "Currently not implemented"
   "low_frequency", 20, "Frequency cut off for feature generation"
   "high_frequency", 7800, "Frequency cut off for feature generation"
   "sample_frequency", 16000, "Sample rate to up- or down-sample to"
   "allow_downsample", True, "Flag for allowing down-sampling"
   "allow_upsample", True, "Flag for allowing up-sampling"
   "splice_left_context", 3, "Frame width for generating LDA transforms"
   "splice_right_context", 3, "Frame width for generating LDA transforms"
   "use_mp", True, "Flag for whether to use multiprocessing feature generation"

.. _training_config:

Training configuration
======================

Global alignment options can be overwritten for each trainer (i.e., different beam settings at different stages of training).

.. note::

   Subsets are created by sorting the utterances by length, taking a larger subset (10 times the specified subset amount)
   and then randomly sampling the specified subset amount from this larger subset.  Utterances with transcriptions that
   are only one word long are ignored.

Monophone Configuration
-----------------------

For the Kaldi recipe that monophone training is based on, see
https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_mono.sh


.. csv-table::
   :header: "Parameter", "Default value", "Notes"

   "subset", 0, "Number of utterances to use (0 uses the full corpus)"
   "num_iterations", 40, "Number of training iterations"
   "max_gaussians", 40, "Total number of gaussians"
   "power", 0.25, "Exponent for gaussians based on occurrence counts"


Realignment iterations for training are calculated based on splitting the number of iterations into quarters.  The first
quarter of training will perform realignment every iteration, the second quarter will perform realignment every other iteration,
and the final two quarters will perform realignment every third iteration.


Triphone Configuration
----------------------

For the Kaldi recipe that triphone training is based on, see
https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_deltas.sh

.. csv-table::
   :header: "Parameter", "Default value", "Notes"

   "subset", 0, "Number of utterances to use (0 uses the full corpus)"
   "num_iterations", 35, "Number of training iterations"
   "max_gaussians", 10000, "Total number of gaussians"
   "power", 0.25, "Exponent for gaussians based on occurrence counts"
   "num_leaves", 1000, "Number of states in the decision tree"
   "cluster_threshold", -1, "Threshold for clustering leaves in decision tree"


LDA Configuration
-----------------

For the Kaldi recipe that LDA training is based on, see
https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_lda_mllt.sh

.. csv-table::
   :header: "Parameter", "Default value", "Notes"

   "subset", 0, "Number of utterances to use (0 uses the full corpus)"
   "num_iterations", 35, "Number of training iterations"
   "max_gaussians", 10000, "Total number of gaussians"
   "power", 0.25, "Exponent for gaussians based on occurrence counts"
   "num_leaves", 1000, "Number of states in the decision tree"
   "cluster_threshold", -1, "Threshold for clustering leaves in decision tree"
   "lda_dimension", 40, "Dimension of resulting LDA features"
   "random_prune", 4.0, "Ratio of random pruning to speed up MLLT"


LDA estimation will be performed every other iteration for the first quarter of iterations, and then one final LDA estimation
will be performed halfway through the training iterations.

Speaker-adapted training (SAT) configuration
--------------------------------------------

For the Kaldi recipe that SAT training is based on, see
https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_sat.sh

.. csv-table::
   :header: "Parameter", "Default value", "Notes"

   "subset", 0, "Number of utterances to use (0 uses the full corpus)"
   "num_iterations", 35, "Number of training iterations"
   "max_gaussians", 10000, "Total number of gaussians"
   "power", 0.25, "Exponent for gaussians based on occurrence counts"
   "num_leaves", 1000, "Number of states in the decision tree"
   "cluster_threshold", -1, "Threshold for clustering leaves in decision tree"
   "silence_weight", 0.0, "Weight on silence in fMLLR estimation"
   "fmllr_update_type", "full", "Type of fMLLR estimation"


fMLLR estimation will be performed every other iteration for the first quarter of iterations, and then one final fMLLR estimation
will be performed halfway through the training iterations.


.. _default_training_config:

Default training config file
----------------------------

.. code-block:: yaml

   beam: 10
   retry_beam: 40

   features:
     type: "mfcc"
     use_energy: false
     frame_shift: 10

   training:
     - monophone:
         num_iterations: 40
         max_gaussians: 1000
         subset: 2000
         boost_silence: 1.25

     - triphone:
         num_iterations: 35
         num_leaves: 2000
         max_gaussians: 10000
         cluster_threshold: -1
         subset: 5000
         boost_silence: 1.25
         power: 0.25

     - lda:
         num_leaves: 2500
         max_gaussians: 15000
         subset: 10000
         num_iterations: 35
         features:
             splice_left_context: 3
             splice_right_context: 3

     - sat:
         num_leaves: 2500
         max_gaussians: 15000
         power: 0.2
         silence_weight: 0.0
         fmllr_update_type: "diag"
         subset: 10000
         features:
             lda: true

     - sat:
         num_leaves: 4200
         max_gaussians: 40000
         power: 0.2
         silence_weight: 0.0
         fmllr_update_type: "diag"
         subset: 30000
         features:
             lda: true
             fmllr: true

.. _1.0_training_config:

Training configuration for 1.0
------------------------------

.. code-block:: yaml

   beam: 10
   retry_beam: 40

   features:
     type: "mfcc"
     use_energy: false
     frame_shift: 10

   training:
     - monophone:
         num_iterations: 40
         max_gaussians: 1000
         boost_silence: 1.0

     - triphone:
         num_iterations: 35
         num_leaves: 3100
         max_gaussians: 50000
         cluster_threshold: 100
         boost_silence: 1.0
         power: 0.25

     - sat:
         num_leaves: 3100
         max_gaussians: 50000
         power: 0.2
         silence_weight: 0.0
         cluster_threshold: 100
         fmllr_update_type: "full"


.. _align_config:

Align configuration
===================

.. code-block:: yaml

   beam: 10
   retry_beam: 40
