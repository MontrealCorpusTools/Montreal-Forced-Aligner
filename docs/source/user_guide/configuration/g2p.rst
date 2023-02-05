
.. _configuration_g2p:

*****************
G2P Configuration
*****************

Global options
==============


.. csv-table::
   :widths: 20, 20, 60
   :header: "Parameter", "Default value", "Notes"
   :escape: '

   "punctuation", "、。।，@<>'"'(),.:;¿?¡!\\&%#*~【】，…‥「」『』〝〟″⟨⟩♪・‹›«»～′$+=", "Characters to treat as punctuation and strip from around words"
   "clitic_markers", "'''’", "Characters to treat as clitic markers, will be collapsed to the first character in the string"
   "compound_markers", "\-", "Characters to treat as marker in compound words (i.e., doesn't need to be preserved like for clitics)"
   "num_pronunciations", 1, "Number of pronunciations to generate"


.. _train_g2p_config:

G2P training options
====================

In addition to the parameters above, the following parameters are used as part of training a G2P model.

.. csv-table::
   :widths: 20, 20, 60
   :header: "Parameter", "Default value", "Notes"

   "order", 7, "Ngram order of the G2P Model"
   "random_starts", 25, "Number of random starts for aligning orthography to phones"
   "seed", 1917, "Seed for randomization"
   "delta", 1/1024, "Comparison/quatization delta for Baum-Welch training"
   "lr", 1.0, "Learning rate for Baum-Welch training"
   "batch_size", 200, "Batch size for Baum-Welch training"
   "max_iterations", 10, "Maximum number of iterations to use in Baum-Welch training"
   "smoothing_method", "kneser_ney", "Smoothing method for the ngram model"
   "pruning_method", "relative_entropy", "Pruning method for pruning the ngram model"
   "model_size", 1000000, "Target number of ngrams for pruning"

Example G2P configuration files
===============================

.. _default_train_g2p_config:

Default G2P training config file
--------------------------------

.. code-block:: yaml

   punctuation: "、。।，@<>\"(),.:;¿?¡!\\&%#*~【】，…‥「」『』〝〟″⟨⟩♪・‹›«»～′$+="
   clitic_markers: "'’"
   compound_markers: "-"
   num_pronunciations: 1  # Used if running in validation mode
   order: 7
   random_starts: 25
   seed: 1917
   delta: 0.0009765
   lr: 1.0
   batch_size: 200
   max_iterations: 10
   smoothing_method: "kneser_ney"
   pruning_method: "relative_entropy"
   model_size: 1000000


.. _default_g2p_config:

Default dictionary generation config file
-----------------------------------------

.. code-block:: yaml

   punctuation: "、。।，@<>\"(),.:;¿?¡!\\&%#*~【】，…‥「」『』〝〟″⟨⟩♪・‹›«»～′$+="
   clitic_markers: "'’"
   compound_markers: "-"
   num_pronunciations: 1
