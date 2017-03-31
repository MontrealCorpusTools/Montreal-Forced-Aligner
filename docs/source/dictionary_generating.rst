.. _dict_generating:

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

```python main.py --language=<DESIRED_LANGUAGE_CODE> --input_dir=<LOCATION_OF_FILES>```

Other options include specififying an output directory and whether the text should be decomposed (to be used if you are working with a Korean dataset, in which case decomposing the Hangul greatly increases the accuracy). All options, as well as language codes, can be viewed by inputting ```final_code_here```.  


Example
=============
In ```MontrealForcedAligner/examples/example_labs``` you will find several sample .lab files from the `THCHS-30 corpus <http://www.openslr.org/18/>`_. These are organized much as they would be for any alignment task. The dictionary reconstructor will read each one of these files, creating a word list, and will be able to build a pronunciation dictionary with a transcription for each one of these words.







