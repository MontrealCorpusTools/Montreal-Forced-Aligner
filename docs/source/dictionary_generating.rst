.. _dict_generating:

.. _`THCHS-30`: http://www.openslr.org/18/



***********************
Generating a Dictionary
***********************

Included with MFA is a separate tool to generate a dictionary from a preexisting model. This should be used if aligned a dataset for which you have no pronunciation dictionary available. The models represent a G2P (Grapheme to Phoneme) system that currently generates word transcriptions in the CMUDict alphabet. These models were made using Phonetisaurus `Phonetisaurus <https://github.com/AdolfVonKleist/Phonetisaurus>`_ and the GlobalPhone dataset. Current language options are: Arabic, Bulgarian, Mandarin, Czech, Polish, Russian, Swahili, Ukrainian, and Vietnamese, with the following accuracies when trained on 90% of the data and tested on 10%:

+----------------------+
| Language  | Accuracy |
+======================+
| Arabic    |   91.8   |
+----------------------+
| Bulgarian |   97.0   |
+----------------------+
| Croatian  |   89.0   |
+----------------------+
| Mandarin  |   99.9   |
+----------------------+
| Czech     |   96.2   |
+----------------------+
| Polish    |   98.7   |
+----------------------+
| Russian   |   96.1   |
+----------------------+
| Swahili   |   99.9   |
+----------------------+
| Ukrainian |   98.7   |
+----------------------+
| Vietnamese|   95.2   |
+----------------------+


All the languages included attain >95% accuracy when used to reconstruct the dictionary from the entire dataset. 


Use
=======

To reconstruct a pronunication dictionary from your .lab or .TextGrid files, simply input: 

```bin/generate_dict --path_to_models=<LOCATION_OF_MODEL> --input_dir=<LOCATION_OF_FILES>```

Other options include specififying an output directory and whether the text should be decomposed (to be used if you are working with a Korean dataset, in which case decomposing the Hangul greatly increases the accuracy). All options, as well as language codes, can be viewed by inputting ```bin/generate_dict --help```.  


Example
=============
In ```MontrealForcedAligner/examples/example_labs``` you will find several sample .lab files from the `THCHS-30`_ corpus. These are organized much as they would be for any alignment task. The dictionary reconstructor will read each one of these files, creating a word list, and will be able to build a pronunciation dictionary with a transcription for each one of these words. Let's start by running the reconstructor, as before: 

```bin/generate_dict --path_to_models=Montreal-Forced-Aligner/examples/CH_models --input_dir=Montreal-Forced-Aligner/examples/CH --outfile=Montreal-Forced-Aligner/examples/chinese_dict.txt```

This should take no more than a few seconds. Open the output file, and check that all the words are there. The accuracy of the transcription should be near 100%. You can now use this to align your mini corpus:

```bin/mfa_train_and_align Montreal-Forced-Aligner/examples/CH  Montreal-Forced-Aligner/examples/chinese_dict.txt Montreal-Forced-Aligner/examples/```

Since there are very few files, the alignment will almost certainly be suboptimal. This example is intended more to give a sense of the pipeline for generating a dictionary and using it for alignment. 




