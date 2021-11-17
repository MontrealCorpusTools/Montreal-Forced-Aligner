
.. _configuration_ivector:

*********************
Ivector Configuration
*********************

For the Kaldi recipe that ivector extractor training is based on, see :kaldi_steps_sid:`train_diag_ubm` and :kaldi_steps_sid:`train_ivector_extractor`.

.. csv-table::
   :header: "Parameter", "Default value", "Notes"

   "ubm_num_iterations", 4, "Number of iterations for training UBM"
   "ubm_num_gselect", 30, "Number of Gaussian-selection indices to use while training"
   "ubm_num_frames", 500000, "Number of frames to keep in memory for initialization"
   "ubm_num_gaussians", 256, ""
   "ubm_num_iterations_init", 20, "Number of iteration to use when initializing UBM"
   "ubm_initial_gaussian_proportion", 0.5, "Start with half the target number of Gaussians"
   "ubm_min_gaussian_weight", 0.0001, ""
   "ubm_remove_low_count_gaussians", True, ""
   "ivector_dimension", 128, "Dimension of extracted ivectors"
   "num_iterations", 10, "Number of training iterations"
   "num_gselect", 20, "Gaussian-selection using diagonal model: number of Gaussians to select"
   "posterior_scale", 1.0, "Scale on posteriors to correct for inter-frame correlation"
   "silence_weight", 0.0, ""
   "min_post", 0.025, "Minimum posterior to use (posteriors below this are pruned out)"
   "num_samples_for_weights", 3, ""
   "gaussian_min_count", 100, ""
   "subsample", 5, "Speeds up training (samples every Xth frame)"
   "max_count", 100, ""
   "apply_cmn", True, "Flag for whether to apply CMVN to input features"


.. _default_ivector_training_config:

Default training config file
----------------------------

.. code-block:: yaml

   features:
     type: "mfcc"
     use_energy: true
     frame_shift: 10

   training:
     - ivector:
         num_iterations: 10
         gaussian_min_count: 2
         silence_weight: 0.0
         posterior_scale: 0.1
         max_count: 100
