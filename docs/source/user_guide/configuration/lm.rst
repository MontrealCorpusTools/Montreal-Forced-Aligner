
.. _configuration_language_modeling:

*******************************
Language model training options
*******************************

See also the :ref:`configuration_dictionary` for the options that control how text is normalized and parsed.


.. csv-table::
   :widths: 20, 20, 60
   :header: "Parameter", "Default value", "Notes"

   "order", 3, "Order of language model"
   "method", kneser_ney, "Method for smoothing"
   "prune_thresh_small", 0.0000003, "Threshold for pruning a small model, only used if ``prune`` is true"
   "prune_thresh_medium", 0.0000001, "Threshold for pruning a medium model, only used if ``prune`` is true"

Default language model config
-----------------------------

.. code-block:: yaml

   order: 3
   method: kneser_ney
   prune_thresh_small: 0.0000003
   prune_thresh_medium: 0.0000001
