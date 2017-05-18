.. _aligning:

.. _`LibriSpeech corpus`: http://www.openslr.org/12/

*******************
Running the aligner
*******************

.. note::

   We assume Unix-style slashes in paths here.  If you're using Windows, change the slashes to backslashes ``\``.

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

.. cmdoption:: -i
               --ignore_exceptions

  Ignore exceptions in loading the corpus. An error will be thrown if two files share the same name, adding this flag will
  ensure utterance names will be unique by appending numbers to the ends of the file names.

Align using pretrained models
-----------------------------

The Montreal Forced Aligner comes with pretrained models/dictionaries for:

- English - trained from the LibriSpeech data set (`LibriSpeech corpus`_)
- Quebec French - coming soon

Command template:

.. code-block:: bash

   bin/mfa_align corpus_directory dictionary_path acoustic_model_path output_directory

.. note::
   ``acoustic_model_path`` can also be a language that has been pretrained ("english" at the moment but other languages coming soon)

Extra options (in addition to the common ones listed above):

.. cmdoption:: -n
               --no_speaker_adaptation

   Flag to disable using speaker adaptation, useful if aligning a small dataset or if speed is more important

.. cmdoption:: -e
               --errors

   Flag for whether utterance transcriptions should be checked for errors prior to aligning

Steps to align:

1. Open terminal or command window, and change directory to ``montreal-forced-aligner`` folder

2. Type ``bin/mfa_align`` followed by the arguments described
   above

.. note::
   On Mac/Unix, to save time typing out the path, you
   can drag a folder from Finder into Terminal and it will put the full
   path to that folder into your command.

   On Windows, you can hold Shift and right-click on a folder/file. Select
   "Copy as path..." and paste it into the command window.



Align using only the data set
-----------------------------


Command template:

.. code-block:: bash

   bin/mfa_train_and_align corpus_directory dictionary_path output_directory


Extra options (in addition to the common ones listed above):

.. cmdoption:: -f
               --fast

  The aligner will do alignment with half the normal amount of iterations

.. cmdoption:: -o PATH
               --output_model_path PATH

  Path to a zip file to save the results' acoustic models (and dictionary)
  from training to use in future aligning

.. cmdoption:: --no_dict

  If this option is specified, the pronunciation for any given word will be
  the orthography, useful for transparent orthographies that have near one-to-one
  correspondence between sounds and alphabet symbols

.. note::

   The arguments ``dictionary_path`` and ``--no_dict`` are mutually exclusive
   and one of the two must be specified to align a data set. Dictionaries can also be generated through using a
   G2P model with the command ``generate_dictionary``.

Steps to align:

1. Open terminal or command window, and change directory to the ``montreal-forced-aligner`` folder

2. Type ``bin/mfa_train_and_align`` followed by the arguments described
   above

An example command:

.. code-block:: bash

   bin/mfa_train_and_align ~/2_French_training ~/French/fr-QuEu.dict ~/2_French_aligned -s 7 -f -v

This command will train a new model and align the files in ``~/2_French_training``
using the dictionary file ``~/French/fr-QuEu.dict``, and save the output
TextGrids to ``~/2_French_training``.  It will take the first 7 characters
of the file name to be the speaker ID number.  It will be fast (do half
as many training iterations) and verbose (output more info to Terminal during training).

Once the aligner finishes, the resulting TextGrids will be in the
specified output directory.  Training can take several hours for large datasets.
