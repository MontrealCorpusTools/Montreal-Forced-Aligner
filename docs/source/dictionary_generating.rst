.. _dict_generating:

.. _`THCHS-30`: http://www.openslr.org/18/
.. _`Phonetisaurus`: https://github.com/AdolfVonKleist/Phonetisaurus



***********************
Generating a dictionary
***********************

Included with MFA is a separate tool to generate a dictionary from a preexisting model. This should be used if you're
aligning a dataset for which you have no pronunciation dictionary or the orthography is very transparent. We have pretrained models for several languages,
which can be downloaded below. These models were generated using Phonetisaurus
`Phonetisaurus`_ and the GlobalPhone dataset. This  means that they will only work for transcriptions which use the same
alphabet. Current language options are: Arabic, Bulgarian, Mandarin, Czech, Polish, Russian, Swahili, Ukrainian,
and Vietnamese, with the following accuracies when trained on 90% of the data and tested on 10%:

+------------+-----------------------------------------------------------------------------+----------+
| Language   | Link                                                                        | Accuracy |
+============+=============================================================================+==========+
| Arabic     | `arabic_g2p <http://mlmlab.org/mfa/mfa-models/g2p/arabic_g2p.zip>`_         |   91.8   |
+------------+-----------------------------------------------------------------------------+----------+
| Bulgarian  | `bulgarian_g2p <http://mlmlab.org/mfa/mfa-models/g2p/bulgarian_g2p.zip>`_   |   97.0   |
+------------+-----------------------------------------------------------------------------+----------+
| Croatian   | `croatian_g2p <http://mlmlab.org/mfa/mfa-models/g2p/croatian_g2p.zip>`_     |   89.0   |
+------------+-----------------------------------------------------------------------------+----------+
| Czech      | `czech_g2p <http://mlmlab.org/mfa/mfa-models/g2p/czech_g2p.zip>`_           |   96.2   |
+------------+-----------------------------------------------------------------------------+----------+
| Mandarin   | `mandarin_g2p <http://mlmlab.org/mfa/mfa-models/g2p/mandarin_g2p.zip>`_     |   99.9   |
+------------+-----------------------------------------------------------------------------+----------+
| Polish     | `polish_g2p <http://mlmlab.org/mfa/mfa-models/g2p/polish_g2p.zip>`_         |   98.7   |
+------------+-----------------------------------------------------------------------------+----------+
| Russian    | `russian_g2p <http://mlmlab.org/mfa/mfa-models/g2p/russian_g2p.zip>`_       |   96.1   |
+------------+-----------------------------------------------------------------------------+----------+
| Swahili    | `swahili_g2p <http://mlmlab.org/mfa/mfa-models/g2p/swahili_g2p.zip>`_       |   99.9   |
+------------+-----------------------------------------------------------------------------+----------+
| Ukrainian  | `ukrainian_g2p <http://mlmlab.org/mfa/mfa-models/g2p/ukrainian_g2p.zip>`_   |   98.7   |
+------------+-----------------------------------------------------------------------------+----------+
| Vietnamese | `vietnamese_g2p <http://mlmlab.org/mfa/mfa-models/g2p/vietnamese_g2p.zip>`_ |   95.2   |
+------------+-----------------------------------------------------------------------------+----------+


Since the point of the model is to reconstruct a dictionary as accurately as possible, they were trained on 100% of the
data, and all attain >95% accuracy when used to reconstruct the dictionary from the entire dataset.

Use
===

To reconstruct a pronunciation dictionary from your .lab or .TextGrid files, simply input:

.. code-block:: bash

    bin/mfa_generate_dict /path/to/model/file.zip /path/to/corpus



Example
=======

In ``examples/example_labs`` you will find several sample .lab files (orthographic transcriptions)
from the `THCHS-30`_ corpus. These are organized much as they would be for any alignment task. The dictionary reconstructor will
create a word list of all the orthographic word-forms in the files, and will build a pronunciation dictionary with a
phonetic transcription for each one of these words, which it will write to a file. Let's start by running the reconstructor, as before:

.. code-block:: bash

   bin/mfa_generate_dict examples/CH_models examples/CH chinese_dict.txt

This should take no more than a few seconds. Open the output file, and check that all the words are there. The accuracy
of the transcription should be near 100%. You can now use this to align your mini corpus:

.. code-block:: bash

   bin/mfa_train_and_align examples/CH  examples/chinese_dict.txt examples/aligned_output

Since there are very few files (i.e. small training set), the alignment will be suboptimal. This example is intended more
to give a sense of the pipeline for generating a dictionary and using it for alignment.



