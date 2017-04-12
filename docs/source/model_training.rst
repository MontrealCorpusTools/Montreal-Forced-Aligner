.. _model_training:

.. _`THCHS-30`: http://www.openslr.org/18/
.. _`Phonetisaurus`: https://github.com/AdolfVonKleist/Phonetisaurus



****************
Training a Model
****************

Another tool included with MFA allows you to train a G2P (Grapheme to Phoneme) model automatically from a given pronunciation dictionary. This type of model can be used for :doc:`generating dictionaries <dictionary_generating>`. It requires a pronunciation dictionary with each line consisting of the orthographic transcription followed by the phonetic transcription. The model is generated using the `Phonetisaurus`_ software, which generates FST (finite state transducer) files. The final fill produced should be called "full.fst". 

Use
=======

Required options
---------------
.. cmdoption:: --path_to_dict
                --path_to_dict PATH
        The user inputs the path to a pronunciation dictionary

.. cmdoption:: --path
                --path PATH
        The user specifies the desired destination for the model

To train a model from a pronunciation dictionary, the following command is used: 

.. code-block:: bash

    bin/train_g2p --path_to_dict=<LOCATION_OF_DICT> --path=<DESTINATION_OF_MODEL>

One optional argument, ```--KO``` is also available. This should be used if working with a Hangul dictionary, as it decomposes the dictionary and increases the accuracy greatly.  All options can be viewed by inputting ```bin/train_g2p --help```.  


Example
=============

In ```Montreal-Forced-Aligner/examples/``` you will find a small Chinese dictionary. It is too small to generate a usable model, but can provide a helpful example. Inputting 

.. code-block:: bash

    bin/train_g2p --path_to_dict=Montreal-Forced-Aligner/examples/chinese_dict.txt --path=Montreal-Forced-Aligner/examples/CH_test_model   

This should take no more than a few seconds, and should produce a model which could be used for :doc:`generating dictionaries <dictionary_generating>` 

**NB** because there is so little data in chinese_dict.txt, the model produced will not be very accurate. Thus any dictionary generated from it will also be inaccurate. 
