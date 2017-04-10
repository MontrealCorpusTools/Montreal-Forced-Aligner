.. _model_training:

.. _`THCHS-30`: http://www.openslr.org/18/



****************
Training a Model
****************

Another tool included with MFA allows the user to train a G2P (Grapheme to Phoneme) model automatically from a given pronunciation dictionary. This type of model can be used for :doc:`generating dictionaries <dictionary_generating>`. It requires a pronunciation dictionary consisting of the orthographic form followed by the pronunciation. The model is generated using the `Phonetisaurus <https://github.com/AdolfVonKleist/Phonetisaurus>`_ software, which generates FST (finite state transducer) files. 

Use
=======

To train a model from a pronunciation dictionary, the following command is used: 

```bin/train_g2p --path_to_dict=<LOCATION_OF_DICT> --path=<DESTINATION_OF_MODEL>```

One optional argument, ```--KO``` is also available. This should be used if working with a Hangul dictionary, as it decomposes the dictionary and increases the accuracy greatly.  All options can be viewed by inputting ```bin/train_g2p --help```.  


Example
=============
In ```MontrealForcedAligner/examples/``` you will find a small Chinese dictionary. It is too small to generate an actually useful model, but can provide an example of the pipeline

```bin/train_g2p --path_to_dict=Montreal-Forced-Aligner/examples/chinese_dict.txt --path=Montreal-Forced-Aligner/examples/CH_test_model```

This should take no more than a few seconds, and should produce a model which can be used for :doc:`generating dictionaries <dictioanry_generating>`
