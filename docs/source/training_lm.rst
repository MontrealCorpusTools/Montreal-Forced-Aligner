.. _training_lm:

************************
Training language models
************************

MFA has a utility function for training ARPA-format ngram language models, as well as merging with a pre-existing model.

Steps to train:

1. Provided the steps in :ref:`installation` have been completed and you are in the same Conda/virtual environment that
   MFA was installed in.

2. Run the following command, substituting the arguments with your own paths:

  .. code-block:: bash

     mfa train_lm corpus_directory output_model_path


Options available:

.. option:: -h
               --help

  Display help message for the command

.. option:: --config_path PATH

   Path to a YAML config file for training the language model. see
   :ref:`train_lm_config` for more details.

.. option:: --model_path PATH

   Path to an existing language model to merge with the training data.

.. option:: --model_weight WEIGHT

   Specify the weight of the supplemental model when merging with the model from the training data.

.. option:: -t DIRECTORY
               --temp_directory DIRECTORY

   Temporary directory root to use for aligning, default is ``~/Documents/MFA``
