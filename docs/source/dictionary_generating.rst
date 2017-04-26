.. _dict_generating:

.. _`THCHS-30`: http://www.openslr.org/18/
.. _`Phonetisaurus`: https://github.com/AdolfVonKleist/Phonetisaurus


***********************
Generating a Dictionary
***********************

Included with MFA is a separate tool to generate a dictionary from a preexisting model. This should be used if you're aligning a dataset for which you have no pronunciation dictionary. We have pretrained models for several languages, which can be downloaded `here <TODO: MAKE WEBPAGE FOR MODEL DOWNLOAD>`_. These models were generated using Phonetisaurus `Phonetisaurus`_ and the GlobalPhone dataset. This  means that they will only work for transcriptions which use the same alphabet (CMUDict). Current language options are: Arabic, Bulgarian, Mandarin, Czech, Polish, Russian, Swahili, Ukrainian, and Vietnamese, with the following accuracies when trained on 90% of the data and tested on 10%:

+-----------+----------+
| Language  | Accuracy |
+===========+==========+
| Arabic    |   91.8   |
+-----------+----------+
| Bulgarian |   97.0   |
+-----------+----------+
| Croatian  |   89.0   |
+-----------+----------+
| Mandarin  |   99.9   |
+-----------+----------+
| Mandarin  |    98.7  | 
| with chars|          |
+-----------+----------+
| Czech     |   96.2   |
+-----------+----------+
| Polish    |   98.7   |
+-----------+----------+
| Russian   |   96.1   |
+-----------+----------+
| Swahili   |   99.9   |
+-----------+----------+
| Ukrainian |   98.7   |
+-----------+----------+
| Vietnamese|   95.2   |
+-----------+----------+


Since the point of the model is to reconstruct a dictionary as accurately as possible, they were trained on 100% of the data, and all attain >95% accuracy when used to reconstruct the dictionary from the entire dataset.

Use
=======

Required options
------------------
.. cmdoption:: --path_to_models
                --path_to_models PATH
        The user inputs the path to generated models or pre-existing ones downloaded from `here <TODO: MAKE WEBPAGE FOR MODEL DOWNLOAD>`_

.. cmdoption:: --input_dir
                --input_dir PATH
        The user specifies the path to the directory containing the transcriptions (whether theyy are .lab or .TextGrid)

Command template 
-----------------
To reconstruct a pronunication dictionary from your .lab or .TextGrid files, simply input: 

.. code-block:: bash

    bin/generate_dict --path_to_models=<LOCATION_OF_MODEL> --input_dir=<LOCATION_OF_FILES>

Other options include specififying an output directory and whether the text should be decomposed (to be used if you are working with a Korean dataset, in which case decomposing the Hangul greatly increases the accuracy). All options can be viewed by inputting ```bin/generate_dict --help```.  


Example
=============

In ```Montreal-Forced-Aligner/examples/CH_chars``` you will find several sample .lab files (orthographic transcriptions) and .wav files (audio) from the `THCHS-30`_ corpus. These are organized much as they would be for any alignment task, and contain the transcriptions (in Hanzi) of the audio in the correspondingly labeled .wav file. The dictionary reconstructor will create a word list of all the orthographic word-forms in the files, and will build a pronunciation dictionary with a phonetic transcription for each one of these words, which it will write to a file. All the following commands assume you are in the MFA home directory. Let's start by running the reconstructor, as before: 

```bin/generate_dict --dict_model_path=examples/CH_models_chars --input_dir=examples/CH_chars --outfile=examples/chinese_dict_char.txt```

This should take no more than a few seconds. Open the output file, and check that all the words are there. The accuracy of the transcription should be near 100%. You can now use this to align your mini corpus:

TODO: MAKE SURE JUST ALIGN IS CORRECT
```bin/mfa_train_and_align examples/CH_chars  examples/chinese_dict_char.txt examples/```

Since there are very few files (i.e. small training set), the alignment will be suboptimal. This example is intended more to give a sense of the pipeline for generating a dictionary and using it for alignment. 




