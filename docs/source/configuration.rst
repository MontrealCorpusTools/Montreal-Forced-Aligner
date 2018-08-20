
.. _configuration:

*************
Configuration
*************

Global options
==============

These options are used for aligning the full dataset (and as part of training).  Increasing the values of them will
allow for more relaxed restrictions on alignment.  Relaxing these restrictions can be particularly helpful for certain
kinds of files that are quite different from the training dataset (i.e., single word production data from experiments,
or longer stretches of audio).

+---------------------------------+----------------------------+-----------------------------------------------------+
|Parameter                        | Default value              |       Notes                                         |
+=================================+============================+=====================================================+
|beam                             | 10                         |  Initial beam width to use for alignment            |
+---------------------------------+----------------------------+-----------------------------------------------------+
|retry_beam                       | 40                         |  Beam width to use if initial alignment fails       |
+---------------------------------+----------------------------+-----------------------------------------------------+
|transition_scale                 | 1.0                        | Multiplier to scale transition costs                |
+---------------------------------+----------------------------+-----------------------------------------------------+
|acoustic_scale                   | 0.1                        | Multiplier to scale acoustic costs                  |
+---------------------------------+----------------------------+-----------------------------------------------------+
|self_loop_scale                  | 0.1                        | Multiplier to scale self loop costs                 |
+---------------------------------+----------------------------+-----------------------------------------------------+
|boost_silence                    | 1.0                        | 1.0 is the value that does not affect probabilities |
+---------------------------------+----------------------------+-----------------------------------------------------+


Feature Configuration
=====================


+---------------------------------+----------------------------+-----------------------------------------------------+
|Parameter                        | Default value              |       Notes                                         |
+=================================+============================+=====================================================+
|type                             | mfcc                       | Currently only MFCCs are supported                  |
+---------------------------------+----------------------------+-----------------------------------------------------+
|use_energy                       | False                      | Use energy in place of first MFCC                   |
+---------------------------------+----------------------------+-----------------------------------------------------+
|frame_shift                      | 10                         | In milliseconds, determines time resolution         |
+---------------------------------+----------------------------+-----------------------------------------------------+


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

+---------------------------------+----------------------------+-----------------------------------------------------+
|Parameter                        | Default value              |       Notes                                         |
+=================================+============================+=====================================================+
|subset                           | 0                          | Number of utterances to use                         |
+---------------------------------+----------------------------+-----------------------------------------------------+
|num_iterations                   | 40                         | Number of training iterations                       |
+---------------------------------+----------------------------+-----------------------------------------------------+
|max_gaussians                    | 1000                       | Total number of gaussians                           |
+---------------------------------+----------------------------+-----------------------------------------------------+
|power                            | 0.25                       | Exponent for gaussians based on occurrence counts   |
+---------------------------------+----------------------------+-----------------------------------------------------+


Realignment iterations for training are calculated based on splitting the number of iterations into quarters.  The first
quarter of training will perform realignment every iteration, the second quarter will perform realignment every other iteration,
and the final two quarters will perform realignment every third iteration.


Triphone Configuration
----------------------

For the Kaldi recipe that triphone training is based on, see
https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_deltas.sh

+---------------------------------+----------------------------+-----------------------------------------------------+
|Parameter                        | Default value              |       Notes                                         |
+=================================+============================+=====================================================+
|subset                           | 0                          | Number of utterances to use                         |
+---------------------------------+----------------------------+-----------------------------------------------------+
|num_iterations                   | 40                         | Number of training iterations                       |
+---------------------------------+----------------------------+-----------------------------------------------------+
|max_gaussians                    | 1000                       | Total number of gaussians                           |
+---------------------------------+----------------------------+-----------------------------------------------------+
|power                            | 0.25                       | Exponent for gaussians based on occurrence counts   |
+---------------------------------+----------------------------+-----------------------------------------------------+
|num_leaves                       | 1000                       | Number of states in the decision tree               |
+---------------------------------+----------------------------+-----------------------------------------------------+
|cluster_threshold                | -1                         | Threshold for clustering leaves in decision tree    |
+---------------------------------+----------------------------+-----------------------------------------------------+


LDA Configuration
-----------------

For the Kaldi recipe that LDA training is based on, see
https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_lda_mllt.sh

+---------------------------------+----------------------------+-----------------------------------------------------+
|Parameter                        | Default value              |       Notes                                         |
+=================================+============================+=====================================================+
|subset                           | 0                          | Number of utterances to use                         |
+---------------------------------+----------------------------+-----------------------------------------------------+
|num_iterations                   | 40                         | Number of training iterations                       |
+---------------------------------+----------------------------+-----------------------------------------------------+
|max_gaussians                    | 1000                       | Total number of gaussians                           |
+---------------------------------+----------------------------+-----------------------------------------------------+
|power                            | 0.25                       | Exponent for gaussians based on occurrence counts   |
+---------------------------------+----------------------------+-----------------------------------------------------+
|num_leaves                       | 1000                       | Number of states in the decision tree               |
+---------------------------------+----------------------------+-----------------------------------------------------+
|cluster_threshold                | -1                         | Threshold for clustering leaves in decision tree    |
+---------------------------------+----------------------------+-----------------------------------------------------+
|lda_dimension                    | 40                         | Dimension of resulting LDA features                 |
+---------------------------------+----------------------------+-----------------------------------------------------+
|random_prune                     | 4.0                        | Ratio of random pruning to speed up MLLT            |
+---------------------------------+----------------------------+-----------------------------------------------------+


LDA estimation will be performed every other iteration for the first quarter of iterations, and then one final LDA estimation
will be performed halfway through the training iterations.

Speaker-adapted training (SAT) configuration
--------------------------------------------

For the Kaldi recipe that SAT training is based on, see
https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_sat.sh

+---------------------------------+----------------------------+-----------------------------------------------------+
|Parameter                        | Default value              |       Notes                                         |
+=================================+============================+=====================================================+
|subset                           | 0                          | Number of utterances to use                         |
+---------------------------------+----------------------------+-----------------------------------------------------+
|num_iterations                   | 40                         | Number of training iterations                       |
+---------------------------------+----------------------------+-----------------------------------------------------+
|max_gaussians                    | 1000                       | Total number of gaussians                           |
+---------------------------------+----------------------------+-----------------------------------------------------+
|power                            | 0.25                       | Exponent for gaussians based on occurrence counts   |
+---------------------------------+----------------------------+-----------------------------------------------------+
|num_leaves                       | 1000                       | Number of states in the decision tree               |
+---------------------------------+----------------------------+-----------------------------------------------------+
|cluster_threshold                | -1                         | Threshold for clustering leaves in decision tree    |
+---------------------------------+----------------------------+-----------------------------------------------------+
|silence_weight                   | 0.0                        |  Weight on silence in fMLLR estimation              |
+---------------------------------+----------------------------+-----------------------------------------------------+
|fmllr_update_type                | "full"                     |  Type of fMLLR estimation                           |
+---------------------------------+----------------------------+-----------------------------------------------------+

fMLLR estimation will be performed every other iteration for the first quarter of iterations, and then one final fMLLR estimation
will be performed halfway through the training iterations.


Default config file
===================

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
         fmllr_power: 0.2
         silence_weight: 0.0
         fmllr_update_type: "diag"
         subset: 10000
         features:
             lda: true

     - sat:
         calc_pron_probs: true
         num_leaves: 4200
         max_gaussians: 40000
         fmllr_power: 0.2
         silence_weight: 0.0
         fmllr_update_type: "diag"
         subset: 30000
         features:
             lda: true
             fmllr: true