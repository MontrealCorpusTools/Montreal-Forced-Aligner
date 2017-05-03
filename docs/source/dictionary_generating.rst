.. _dict_generating:

.. _`THCHS-30`: http://www.openslr.org/18/
.. _`Phonetisaurus`: https://github.com/AdolfVonKleist/Phonetisaurus


***********************
Generating a Dictionary
***********************


Included with MFA is a separate tool to generate a dictionary from a preexisting model. This should be used if you're
aligning a dataset for which you have no pronunciation dictionary. We have pretrained models for several languages,
which can be downloaded `here <http://mlmlab.org/mfa/>`_. These models were generated using Phonetisaurus
`Phonetisaurus`_ and the GlobalPhone dataset. This  means that they will only work for transcriptions which use the same
alphabet. Current language options are: Arabic, Bulgarian, Mandarin, Czech, Polish, Russian, Swahili, Ukrainian,
and Vietnamese, with the following accuracies when trained on 90% of the data and tested on 10%:

+----------------------+--------------------------------------------------------------------+----------+
| Language             | Link                                                               | Accuracy |
+======================+====================================================================+==========+
| Arabic               | `model <http://mlmlab.org/mfa/pretrained/g2p/arabic_g2p.zip>`_     | 91.8     |  
+----------------------+--------------------------------------------------------------------+----------+
| Bulgarian            | `model <http://mlmlab.org/mfa/pretrained/g2p/bulgarian_g2p.zip>`_  |   97.0   |
+----------------------+--------------------------------------------------------------------+----------+
| Croatian             | `model <http://mlmlab.org/mfa/pretrained/g2p/croatian_g2p.zip>`_   |   89.0   |
+----------------------+--------------------------------------------------------------------+----------+
| Czech                | `model <http://mlmlab.org/mfa/pretrained/g2p/czech_g2p.zip>`_      |   96.2   |
+----------------------+--------------------------------------------------------------------+----------+
| Mandarin             | `model <http://mlmlab.org/mfa/pretrained/g2p/mandarin_g2p.zip>`_   |   99.9   |
+----------------------+--------------------------------------------------------------------+----------+
| Polish               | `model <http://mlmlab.org/mfa/pretrained/g2p/polish_g2p.zip>`_     |   98.7   |
+----------------------+--------------------------------------------------------------------+----------+
| Russian              | `model <http://mlmlab.org/mfa/pretrained/g2p/russian_g2p.zip>`_    |   96.1   |
+----------------------+--------------------------------------------------------------------+----------+
| Swahili              | `model <http://mlmlab.org/mfa/pretrained/g2p/swahili_g2p.zip>`_    |   99.9   |
+----------------------+--------------------------------------------------------------------+----------+
| Ukrainian            | `model <http://mlmlab.org/mfa/pretrained/g2p/ukrainian_g2p.zip>`_  |   98.7   |
+----------------------+--------------------------------------------------------------------+----------+
| Vietnamese           | `model <http://mlmlab.org/mfa/pretrained/g2p/vietnamese_g2p.zip>`_ |   95.2   |
+----------------------+--------------------------------------------------------------------+----------+


Since the point of the model is to reconstruct a dictionary as accurately as possible, they were trained on 100% of the data, and all attain >95% accuracy when used to reconstruct the dictionary from the entire dataset.

Use
===

Required options
----------------
.. cmdoption:: g2p_model_path PATH

   The user inputs the path to generated models or pre-existing ones downloaded from `here <http://mlmlab.org/mfa/>`_

.. cmdoption:: corpus_directory PATH

   The user specifies the path to the directory containing the transcriptions (whether they are .lab or .TextGrid)

Command template 
----------------
To reconstruct a pronunciation dictionary from your .lab or .TextGrid files, simply input:

.. code-block:: bash

    bin/generate_dict /path/to/model/file.zip /path/to/corpus

Other options include specifying an output directory and whether the text should be decomposed (to be used if you are
working with a Korean dataset, in which case decomposing the Hangul greatly increases the accuracy). All options can be
viewed by inputting ``bin/generate_dict --help``.


Example
=======

In ``examples/example_labs`` you will find several sample .lab files (orthographic transcriptions)
from the `THCHS-30`_ corpus. These are organized much as they would be for any alignment task. The dictionary reconstructor will
create a word list of all the orthographic word-forms in the files, and will build a pronunciation dictionary with a
phonetic transcription for each one of these words, which it will write to a file. Let's start by running the reconstructor, as before:

.. code-block:: bash

   bin/generate_dict examples/CH_models examples/CH chinese_dict.txt

This should take no more than a few seconds. Open the output file, and check that all the words are there. The accuracy
of the transcription should be near 100%. You can now use this to align your mini corpus:

.. code-block:: bash

   bin/mfa_train_and_align examples/CH  examples/chinese_dict.txt examples/aligned_output

Since there are very few files (i.e. small training set), the alignment will be suboptimal. This example is intended more
to give a sense of the pipeline for generating a dictionary and using it for alignment.





