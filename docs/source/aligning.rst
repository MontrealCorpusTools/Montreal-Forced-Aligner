.. _aligning:

.. _`LibriSpeech corpus`: http://www.openslr.org/12/

*******************
Running the aligner
*******************

.. note::

   We assume Unix-style slashes in paths here.  If you're using Windows, change the slashes ``/`` to backslashes ``\``.

Common options for both aligner executables
-------------------------------------------

.. cmdoption:: -s NUMBER
               --speaker_characters NUMBER

   Number of characters to use to identify speakers; if not specified,
   the aligner assumes that the directory name is the identifier for the
   speaker.  Additionally, it accepts the value ``prosodylab`` to use the second field of a ``_`` delimited file name,
   following the convention of labelling production data in the ProsodyLab at McGill.

.. cmdoption:: -t DIRECTORY
               --temp_directory DIRECTORY

   Temporary directory root to use for aligning, default is ``~/Documents/MFA``

.. cmdoption:: -j NUMBER
               --num_jobs NUMBER

  Number of jobs to use; defaults to 3, set higher if you have more
  processors available and would like to align faster

.. cmdoption:: -v
               --verbose

  The aligner will print out more debugging information if present

.. cmdoption:: -c
               --clean

  Temporary files in ``~/Documents/MFA`` and the output directory will be
  removed prior to aligning.  This is good to use when aligning a new dataset,
  but it shares a name with a previously aligned dataset.

.. cmdoption:: -h
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

   Do not specify an existing directory as the output directory (unless it is from an earlier run of the aligner).  The
   current functionality of the aligner destroys the output directory prior to generating TextGrids.  Future versions will
   be smarter about cleaning up TextGrids from previous runs without removing the directory.

.. note::
   ``acoustic_model_path`` can also be a language that has been pretrained: ``english`` currently works for the English
   acoustic model trained on the `Librispeech corpus`_.

Extra options (in addition to the common ones listed above):


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

   Do not specify an existing directory as the output directory (unless it is from an earlier run of the aligner).  The
   current functionality of the aligner destroys the output directory prior to generating TextGrids.  Future versions will
   be smarter about cleaning up TextGrids from previous runs without removing the directory.


Extra options (in addition to the common ones listed above):

.. cmdoption:: -o PATH
               --output_model_path PATH

  Path to a zip file to save the results' acoustic models (and dictionary)
  from training to use in future aligning

.. note::

   The arguments ``dictionary_path`` and ``--no_dict`` are mutually exclusive
   and one of the two must be specified to align a data set. Dictionaries can also be generated through using a
   G2P model with the command ``generate_dictionary``.

Once the aligner finishes, the resulting TextGrids will be in the
specified output directory.  Training can take several hours for large datasets.
