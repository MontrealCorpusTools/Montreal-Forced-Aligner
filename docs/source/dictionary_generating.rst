

.. _`THCHS-30`: http://www.openslr.org/18/

.. _`example Mandarin corpus`: http://mlmlab.org/mfa/CH_g2p_example.zip

.. _`Mandarin pinyin G2P model`: http://mlmlab.org/mfa/mfa-models/g2p/mandarin_pinyin_g2p.zip


.. _dict_generating:

***********************
Generating a dictionary
***********************

We have trained several G2P models that are available for download (:ref:`pretrained_g2p`).

Use
===

To reconstruct a pronunciation dictionary from your .lab or .TextGrid files, simply input:

.. code-block:: bash

    bin/mfa_generate_dictionary /path/to/model/file.zip /path/to/corpus



Example
=======

Download the `example Mandarin corpus`_ and the `Mandarin pinyin G2P model`_ to any directory. In ``examples/CH`` you will find several sample .lab files (orthographic transcriptions)
from the `THCHS-30`_ corpus. These are organized much as they would be for any alignment task. The dictionary reconstructor will
create a word list of all the orthographic word-forms in the files, and will build a pronunciation dictionary with a
phonetic transcription for each one of these words, which it will write to a file. Let's start by running the reconstructor, as before:

.. code-block:: bash

   bin/mfa_generate_dictionary path/to/mandarin_pinyin_g2p.zip path/to/examples/CH chinese_dict.txt

This should take no more than a few seconds. Open the output file, and check that all the words are there. The accuracy
of the transcription should be near 100%. You can now use this to align your mini corpus:

.. code-block:: bash

   bin/mfa_train_and_align examples/CH  path/to/examples/chinese_dict.txt path/to/examples/aligned_output

Since there are very few files (i.e. small training set), the alignment will be suboptimal. This example is intended more
to give a sense of the pipeline for generating a dictionary and using it for alignment.




