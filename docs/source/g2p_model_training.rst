.. _`Pynini`: https://github.com/kylebgormon/Pynini
.. _`Sigmorphon 2020 G2P task baseline`: https://github.com/sigmorphon/2020/tree/master/task1/baselines/fst

.. _g2p_model_training:

************************
Training a new G2P model
************************

Another tool included with MFA allows you to train a G2P (Grapheme to Phoneme) model automatically from a given
pronunciation dictionary.
This type of model can be used for :ref:`g2p_dictionary_generating`.
It requires a pronunciation dictionary with each line consisting of the orthographic transcription followed by the
phonetic transcription. The model is generated using the `Pynini`_ package, which generates FST (finite state transducer)
files. The implementation is based on that in the `Sigmorphon 2020 G2P task baseline`_.
The G2P model output will be a .zip file like the acoustic model generated from alignment.

To train a model from a pronunciation dictionary, the following command is used:

.. code-block:: bash

    mfa train_g2p dictionary_path output_model_path

The ``dictionary_path`` should be a full path to a pronunciation dictionary to train the model from.  The
``output_model_path`` is the path to save the resulting G2P model.

Extra options:

.. option:: -t DIRECTORY
               --temp_directory DIRECTORY

   Temporary directory root to use for training, default is ``~/Documents/MFA``

.. option:: -j NUMBER
               --num_jobs NUMBER

  Number of jobs to use; defaults to 3, set higher if you have more
  processors available and would like to train the G2P model faster

.. cmdoption:: --order ORDER

   Defines the ngram model order, defaults to 7

.. cmdoption:: -v
               --validate

   Run a validation on the dictionary with 90% of the data as training and 10% as test.  It will output the percentage
   accuracy of pronunciations generated.

.. option:: -c
               --clean

  Forces removal of temporary files under ``~/Documents/MFA`` or the specified temporary directory
  prior to training the model.

.. note::

   See :ref:`g2p_model_training_example` for an example of how to train a G2P model with a premade toy example.
