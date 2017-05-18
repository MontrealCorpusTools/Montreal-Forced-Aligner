

.. _`THCHS-30`: http://www.openslr.org/18/


.. _dict_generating:

***********************
Generating a dictionary
***********************

We have trained several G2P models that are available for download (:ref:`pretrained_g2p`).

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




