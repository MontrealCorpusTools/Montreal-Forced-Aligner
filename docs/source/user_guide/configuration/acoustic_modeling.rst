
.. _configuration_acoustic_modeling:

*******************************
Acoustic model training options
*******************************

.. note::

   See :ref:`configuration_global` for options relating to the alignment steps

Global alignment options can be overwritten for each trainer (i.e., different beam settings at different stages of training).

.. note::

   Subsets are created by sorting the utterances by length, taking a larger subset (10 times the specified subset amount) and then randomly sampling the specified subset amount from this larger subset.  Utterances with transcriptions that are only one word long are ignored.

Monophone Configuration
-----------------------

For the Kaldi recipe that monophone training is based on, see :kaldi_steps:`train_mono`.


.. csv-table::
   :header: "Parameter", "Default value", "Notes"

   "subset", 0, "Number of utterances to use (0 uses the full corpus)"
   "num_iterations", 40, "Number of training iterations"
   "max_gaussians", 40, "Total number of gaussians"
   "power", 0.25, "Exponent for gaussians based on occurrence counts"


Realignment iterations for training are calculated based on splitting the number of iterations into quarters.  The first
quarter of training will perform realignment every iteration, the second quarter will perform realignment every other iteration,
and the final two quarters will perform realignment every third iteration.


Triphone training options
-------------------------

For the Kaldi recipe that triphone training is based on, see :kaldi_steps:`train_deltas`.

.. csv-table::
   :header: "Parameter", "Default value", "Notes"

   "subset", 0, "Number of utterances to use (0 uses the full corpus)"
   "num_iterations", 35, "Number of training iterations"
   "max_gaussians", 10000, "Total number of gaussians"
   "power", 0.25, "Exponent for gaussians based on occurrence counts"
   "num_leaves", 1000, "Number of states in the decision tree"
   "cluster_threshold", -1, "Threshold for clustering leaves in decision tree"


LDA training options
--------------------

For the Kaldi recipe that LDA training is based on, see:kaldi_steps:`train_lda_mllt`.

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

Speaker-adapted training (SAT) options
--------------------------------------

For the Kaldi recipe that SAT training is based on, see:kaldi_steps:`train_sat`.

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

The below configuration file shows the equivalent of the current 2.0 training regime, mostly as an example of what configuration options are available and how they progress through the overall training.

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
         fmllr_update_type: "full"
         subset: 10000

     - sat:
         num_leaves: 4200
         max_gaussians: 40000
         power: 0.2
         silence_weight: 0.0
         fmllr_update_type: "full"
         subset: 30000

.. _1.0_training_config:

Training configuration for 1.0
------------------------------

The below configuration matches the training procedure used in models trained in version 1.0.  Note the lack of an LDA block, and only one SAT training block, as well as the lack of subsets in initial training blocks to speed up overall training.

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
