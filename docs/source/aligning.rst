.. _aligning:

.. _`LibriSpeech corpus`: http://www.openslr.org/12/

*******************
Running the aligner
*******************

Common options for both aligner executables
-------------------------------------------

.. option:: --config_path PATH

   Path to a YAML config file that will specify either the alignment options or the training configuration. see
   :ref:`configuration` for more details.

.. option:: -s NUMBER
               --speaker_characters NUMBER

   Number of characters to use to identify speakers; if not specified,
   the aligner assumes that the directory name is the identifier for the
   speaker.  Additionally, it accepts the value ``prosodylab`` to use the second field of a ``_`` delimited file name,
   following the convention of labelling production data in the ProsodyLab at McGill.

.. option:: -t DIRECTORY
               --temp_directory DIRECTORY

   Temporary directory root to use for aligning, default is ``~/Documents/MFA``

.. option:: -j NUMBER
               --num_jobs NUMBER

  Number of jobs to use; defaults to 3, set higher if you have more
  processors available and would like to align faster

.. option:: -v
               --verbose

  The aligner will print out more information if present

.. option:: -d
               --debug

  The aligner will run in debug mode

.. option:: -c
               --clean

  Forces removal of temporary files in ``~/Documents/MFA``
  prior to aligning.  This is good to use when aligning a new dataset,
  but it shares a name with a previously aligned dataset.  Cleaning automatically happens if the previous alignment
  run had an error.

.. option:: -h
               --help

  Display help message for the executable

Align using pretrained models
-----------------------------

The Montreal Forced Aligner comes with :ref:`pretrained_acoustic` for several languages.

Steps to align:

1. Open Terminal (Mac) or a Command Window (Windows), and change the directory to the root of where you installed the Montreal Forced Aligner:

  .. code-block:: bash

   cd path/to/montreal-forced-aligner/

2. Run the following command, substituting the arguments with your own paths:

  .. code-block:: bash

     bin/mfa_align corpus_directory dictionary_path acoustic_model_path output_directory


.. warning::

   Aligned TextGrids will overwrite any existing TextGrids (with the same name as the wav files) in the output directory.
   The aligner will throw an error if the corpus directory is specified as the output directory (to prevent overwriting
   any input TextGrids).

.. note::

   ``acoustic_model_path`` can also be a language that has been pretrained: ``english`` works by default to specify the English
   acoustic model trained on the `Librispeech corpus`_.  Custom pretrained models can be placed in the ``montreal-forced-aligner/pretrained_models``
   to allow using them with just their name (without the .zip extension).  I.e., downloading ``german.zip`` and putting it there
   would allow you to use ``german`` to specify the acoustic model.

.. note::
   On Mac/Unix, to save time typing out the path, you
   can drag a folder from Finder into Terminal and it will put the full
   path to that folder into your command.

   On Windows, you can hold Shift and right-click on a folder/file. Select
   "Copy as path..." and paste it into the command window.

Once the aligner finishes, the resulting TextGrids will be in the
specified output directory.

Align using only the data set
-----------------------------

Steps to align:

1. Open Terminal (Mac) or a Command Window (Windows), and change the directory to the root of where you installed the
Montreal Forced Aligner:

  .. code-block:: bash

   cd path/to/montreal-forced-aligner/

2. Run the following command, substituting the arguments with your own paths:

  .. code-block:: bash

     bin/mfa_train_and_align corpus_directory dictionary_path output_directory

.. warning::

   Aligned TextGrids will overwrite any existing TextGrids (with the same name as the wav files) in the output directory.
   The aligner will throw an error if the corpus directory is specified as the output directory (to prevent overwriting
   any input TextGrids).


Extra options (in addition to the common ones listed above):

.. option:: -o PATH
               --output_model_path PATH

  Path to a zip file to save the results' acoustic models
  from training to use in future aligning

Once the aligner finishes, the resulting TextGrids will be in the
specified output directory.  Training can take several hours for large datasets.
