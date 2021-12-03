
.. _configuration_adapting:

Acoustic model adaptation options
=================================

For the Kaldi recipe that monophone training is based on, see :kaldi_steps:`train_map`.


.. csv-table::
   :header: "Parameter", "Default value", "Notes"

   "mapping_tau", 20, "smoothing constant used in MAP estimation, corresponds to the number of 'fake counts' that we add for the old model.  Larger tau corresponds to less aggressive re-estimation, and more smoothing.  You might also want to try 10 or 15."
