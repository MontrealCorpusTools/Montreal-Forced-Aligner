

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


.. note::

   See :ref:`dict_generating_example` for an example of how to use G2P functionality with a premade example.





