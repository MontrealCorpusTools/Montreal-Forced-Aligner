
.. _configuration_segmentation:

********************
Segmentation options
********************


.. csv-table::
   :widths: 20, 20, 60
   :header: "Parameter", "Default value", "Notes"

   "energy_threshold", 5.5, "Energy threshold above which a frame will be counted as voiced"
   "energy_mean_scale", 0.5, "Proportion of the mean energy of the file that should be added to the energy_threshold"
   "max_segment_length", 30, "Maximum length of segments before they do not get merged"
   "min_pause_duration", 0.05, "Minimum unvoiced duration to split speech segments"

.. _default_segment_config:

Default segmentation config file
--------------------------------

.. code-block:: yaml

   energy_threshold: 5.5
   energy_mean_scale: 0.5
   max_segment_length: 30
   min_pause_duration: 0.05
