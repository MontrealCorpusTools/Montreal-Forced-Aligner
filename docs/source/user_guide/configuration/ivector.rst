
.. _configuration_ivector:

**********************************
Ivector extractor training options
**********************************

.. warning::

   The current implementation of ivectors is a little spotty and there is a planned pass over the speaker diarization on the roadmap for 2.1.

Diagonal UBM training
=====================

For the Kaldi recipe that DUBM training is based on, see :kaldi_steps_sid:`train_diag_ubm`.

.. csv-table::
   :widths: 20, 20, 60
   :header: "Parameter", "Default value", "Notes"

   "num_iterations", 4, "Number of iterations for training UBM"
   "num_gselect", 30, "Number of Gaussian-selection indices to use while training"
   "subsample", 5, "Subsample factor for feature frames"
   "num_frames", 500000, "Number of frames to keep in memory for initialization"
   "num_gaussians", 256, "Number of gaussians to use for DUBM training"
   "num_iterations_init", 20, "Number of iteration to use when initializing UBM"
   "initial_gaussian_proportion", 0.5, "Start with half the target number of Gaussians"
   "min_gaussian_weight", 0.0001, ""
   "remove_low_count_gaussians", True, "Flag for removing low count gaussians in the final round of training"


Ivector training
================

For the Kaldi recipe that ivector training is based on, see :kaldi_steps_sid:`train_ivector_extractor`.

.. csv-table::
   :widths: 20, 20, 60
   :header: "Parameter", "Default value", "Notes"

   "ivector_dimension", 128, "Dimension of extracted ivectors"
   "num_iterations", 10, "Number of training iterations"
   "num_gselect", 20, "Gaussian-selection using diagonal model: number of Gaussians to select"
   "posterior_scale", 1.0, "Scale on posteriors to correct for inter-frame correlation"
   "silence_weight", 0.0, "Weight of silence in calculating posteriors for ivector extraction"
   "min_post", 0.025, "Minimum posterior to use (posteriors below this are pruned out)"
   "gaussian_min_count", 100, ""
   "subsample", 5, "Speeds up training (samples every Xth frame)"
   "max_count", 100, "The use of this option can make iVectors more consistent for different lengths of utterance, by scaling up the prior term when the data-count exceeds this value. The data-count is after posterior-scaling, so assuming the posterior-scale is 0.1, max_count=100 starts having effect after 1000 frames, or 10 seconds of data."
   "uses_cmvn", True, "Flag for whether to apply CMVN to input features"

.. _default_ivector_training_config:

Default training config file
----------------------------

The below configuration file shows the equivalent of the current 2.0 training regime, mostly as an example of what configuration options are available and how they progress through the overall training.

.. code-block:: yaml

   features:
     type: "mfcc"
     use_energy: true
     frame_shift: 10

   training:
     - dubm:
         num_iterations: 4
         num_gselect: 30
         num_gaussians: 256
         num_iterations_init: 20
     - ivector:
         ivector_dimension: 128
         num_iterations: 10
         gaussian_min_count: 100
         silence_weight: 0.0
         posterior_scale: 0.1
         max_count: 100
