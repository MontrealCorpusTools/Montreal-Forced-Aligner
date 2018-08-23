

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

To construct a pronunciation dictionary from your .lab or .TextGrid files, simply input:

.. code-block:: bash

    bin/mfa_generate_dictionary /path/to/model/file.zip /path/to/corpus /path/to/save

In addition to parsing a corpus ready for alignment, dictionaries can also be generated from simple text files (i.e., one
orthography per line):

.. code-block:: bash

    bin/mfa_generate_dictionary /path/to/model/file.zip /path/to/text/file /path/to/save

.. note::

   This functionality is particularly useful if you would like to generate pronunciations to supplement your existing pronunciation
   dictionary.  Simply run the validation utility (see :ref:`running_the_validator`), and then use the path to the ``oovs_found.txt``
   file that it generates.


Pronunciation dictionaries can also be generated from the orthographies of the words themselves, rather than relying on
a trained G2P model.  This functionality should be reserved for languages with transparent orthographies, close to 1-to-1
grapheme-to-phoneme mapping.

.. code-block:: bash

    bin/mfa_generate_dictionary /path/to/corpus/or/text/file /path/to/save


Example
=======

Download the `example Mandarin corpus`_ and the `Mandarin pinyin G2P model`_ to some place on your machine. In ``examples/CH`` you will find several sample .lab files (orthographic transcriptions)
from the `THCHS-30`_ corpus. These are organized much as they would be for any alignment task. The dictionary reconstructor will
create a word list of all the orthographic word-forms in the files, and will build a pronunciation dictionary with a
phonetic transcription for each one of these words, which it will write to a file. Let's start by running the reconstructor, as before:

.. code-block:: bash

   bin/mfa_generate_dictionary path/to/mandarin_pinyin_g2p.zip path/to/examples/CH path/to/examples/CH chinese_dict.txt

This should take no more than a few seconds. Open the output file, and check that all the words are there. The accuracy
of the transcription should be near 100%. You can now use this to align your mini corpus:

.. code-block:: bash

   bin/mfa_train_and_align path/to/examples/CH  path/to/examples/chinese_dict.txt examples/aligned_output

Since there are very few files (i.e. small training set), the alignment will be suboptimal. This example is intended more
to give a sense of the pipeline for generating a dictionary and using it for alignment.




