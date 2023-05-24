

.. _first_steps:

***********
First steps
***********

The ``mfa`` command line utility has grown over the years to encompass a number of utility functions.  This section aims to provide a path for first-time users to figure out the workflow that works best for them.

Also check out :ref:`tutorials` for external tutorials or blog posts on specific topics.

Use cases
=========

There are several broad use cases that you might want to use MFA for.  Take a look below and if any are close matches, you should be able to apply the linked instructions to your data.

#. **Use case 1:** You have a :ref:`speech corpus <corpus_structure>`, the language has a :xref:`pretrained acoustic model <pretrained_acoustic_models>` and :xref:`pretrained dictionary <pretrained_dictionaries>`.

   #. Follow :ref:`first_steps_align_pretrained` to generate aligned TextGrids

#. **Use case 2:** You have a :ref:`speech corpus <corpus_structure>`, the language has a :xref:`pretrained acoustic model <pretrained_acoustic_models>` and :xref:`pretrained dictionary <pretrained_dictionaries>`, but the coverage of the dictionary for your corpus is not great, but the language has a :xref:`pretrained G2P model <pretrained_g2p>`.

   #. Follow :ref:`first_steps_g2p_oovs` to generate pronunciations for OOV words in the corpus
   #. Use the generated dictionary in :ref:`first_steps_align_pretrained` to generate aligned TextGrids

#. **Use case 3:** You have a :ref:`speech corpus <corpus_structure>`, the language has a :xref:`pretrained acoustic model <pretrained_acoustic_models>` and :xref:`pretrained G2P model <pretrained_g2p>`, but it doesn't have a :xref:`pretrained dictionary <pretrained_dictionaries>`.

   #. Follow :ref:`first_steps_g2p_pretrained` to generate a dictionary
   #. Use the generated dictionary in :ref:`first_steps_align_pretrained` to generate aligned TextGrids

#. **Use case 4:** You have a :ref:`speech corpus <corpus_structure>` and your own :ref:`pronunciation dictionary <dictionary_format>`, but there is no :xref:`pretrained acoustic model <pretrained_acoustic_models>` for the language (or none that have the same phones as the pronunciation dictionary).

   #. Follow :ref:`first_steps_align_train_acoustic_model` to generate aligned TextGrids

#. **Use case 5:** You have a :ref:`speech corpus <corpus_structure>` and your own :ref:`pronunciation dictionary <dictionary_format>`, but it does not have great coverage of the words in the corpus.

   #. Follow :ref:`first_steps_train_g2p` to train a G2P model
   #. Use the trained G2P model in :ref:`first_steps_g2p_pretrained` to generate a pronunciation dictionary
   #. Use the generated pronunciation dictionary in :ref:`first_steps_align_train_acoustic_model` to generate aligned TextGrids

#. **Use case 6:** You have a :ref:`speech corpus <corpus_structure>` and the language has a :xref:`pretrained acoustic model <pretrained_acoustic_models>`, but the language does not mark word boundaries in its orthography (and the language has a :xref:`pretrained tokenizer model <pretrained_tokenizer_models>`).

   #. Follow :ref:`first_steps_tokenize` to tokenize the corpus
   #. Use the tokenized transcripts and follow :ref:`first_steps_align_pretrained`

.. _first_steps_align_pretrained:

Aligning a speech corpus with existing pronunciation dictionary and acoustic model
----------------------------------------------------------------------------------

For the purposes of this example, we'll use the "english_us_arpa" model, but the instructions will be applicable to any pretrained acoustic model/pronunciation dictionary pairing. We'll also assume that you have done nothing else with MFA other than follow the :ref:`installation` instructions and you have the :code:`mfa` command working.  Finally, we'll assume that your :ref:`speech corpus <corpus_structure>` is stored in the folder :code:`~/mfa_data/my_corpus`, so when working with your data, this will be the main thing to update.

First we'll need the pretrained models and dictionary.  These are installed via the :code:`mfa model download` command:

.. code-block::

   mfa model download acoustic english_us_arpa
   mfa model download dictionary english_us_arpa

You should be able to run :code:`mfa model inspect acoustic english_us_arpa` and it will output information about the :code:`english_us_arpa` acoustic model.

Next, we want to make sure that the dataset is in the proper format for MFA, which is what the :code:`mfa validate` command does:

.. code-block::

   mfa validate ~/mfa_data/my_corpus english_us_arpa english_us_arpa

This command will look through the corpus and make sure that MFA is parsing everything correctly.  There are couple of different types of :ref:`corpus_structure` that MFA supports, but in general the core requirement is that you should have pairs of sound files and transcription files with the same name (except for the extension).  Take a look over the validator output and make sure that the number of speakers and number of files and utterances match your expectations, and that the number of Out of Vocabulary (OOV) items is not too high.  If you want to generate transcriptions for these words so that they can be aligned, see :ref:`first_steps_g2p_pretrained` to make a new dictionary.  The validator will also attempt to run feature generation and train a simple monophone model to make sure that everything works within Kaldi.

Once we've validated the data, we can align it via the :code:`mfa align` command:

.. code-block::

   mfa align ~/mfa_data/my_corpus english_us_arpa english_us_arpa ~/mfa_data/my_corpus_aligned

If alignment is successful, you'll see TextGrid files containing the aligned words and phones in the output directory (here :code:`~/mfa_data/my_corpus_aligned`). If there were issues in exporting the TextGrids, you'll see them listed in the output directory.  If your corpus is large, you'll likely want to increase the number of jobs that MFA uses.  For that and more advanced configuration, see :ref:`pretrained_alignment`.

.. note::

   Please see :ref:`alignment_example` for an example using toy data.


.. _first_steps_g2p_pretrained:

Generating a pronunciation dictionary with a pretrained G2P model
-----------------------------------------------------------------

For the purposes of this example, we'll use the "english_us_arpa" model, but the instructions will be applicable to any pretrained G2P model. We'll also assume that you have done nothing else with MFA other than follow the :ref:`installation` instructions and you have the :code:`mfa` command working.  Finally, we'll assume that your corpus is stored in the folder :code:`~/mfa_data/my_corpus`, so when working with your data, this will be the main thing to update.

First we'll need the pretrained G2P model.  These are installed via the :code:`mfa model download` command:

.. code-block::

   mfa model download g2p english_us_arpa

You should be able to run :code:`mfa model inspect g2p english_us_arpa` and it will output information about the :code:`english_us_arpa` G2P model.

Depending on your use case, you might have a list of words to run G2P over, or just a corpus of sound and transcription files.  The :code:`mfa g2p` command can process either:

.. code-block::

   mfa g2p ~/mfa_data/my_corpus english_us_arpa ~/mfa_data/new_dictionary.txt  # If using a corpus
   mfa g2p ~/mfa_data/my_word_list.txt english_us_arpa ~/mfa_data/new_dictionary.txt  # If using a word list

Running one of the above will output a text file pronunciation dictionary in the :ref:`MFA dictionary format <dictionary_format>`.  I recommend looking over the pronunciations generated and make sure that they look sensible.  For languages where the orthography is not transparent, it may be helpful to include :code:`--num_pronunciations 3` so that more pronunciations are generated than just the most likely one. For more details on running G2P, see :ref:`g2p_dictionary_generating`.

From here you can use this dictionary file as input to any MFA command that uses dictionaries, i.e.

.. code-block::

   mfa align ~/mfa_data/my_corpus ~/mfa_data/new_dictionary.txt english_us_arpa ~/mfa_data/my_corpus_aligned


.. note::

   Please see :ref:`dict_generating_example` for an example using toy data.


.. _first_steps_g2p_oovs:

Generating pronunciations for OOV items in a corpus
---------------------------------------------------

For the purposes of this example, we'll use the "english_us_arpa" model, but the instructions will be applicable to any pretrained G2P model. We'll also assume that you have done nothing else with MFA other than follow the :ref:`installation` instructions and you have the :code:`mfa` command working.  Finally, we'll assume that your corpus is stored in the folder :code:`~/mfa_data/my_corpus`, so when working with your data, this will be the main thing to update.

First we'll need the pretrained G2P model.  These are installed via the :code:`mfa model download` command:

.. code-block::

   mfa model download g2p english_us_arpa

You should be able to run :code:`mfa model inspect g2p english_us_arpa` and it will output information about the :code:`english_us_arpa` G2P model.

Depending on your use case, you might have a list of words to run G2P over, or just a corpus of sound and transcription files.  The :code:`mfa g2p` command can process either:

.. code-block::

   mfa g2p ~/mfa_data/my_corpus english_us_arpa ~/mfa_data/g2pped_oovs.txt --dictionary_path english_us_arpa

Running the above will output a text file in the format that MFA uses (:ref:`dictionary_format`) with all the OOV words (ignoring bracketed words like :ipa_inline:`<cutoff>`).  I recommend looking over the pronunciations generated and make sure that they look sensible.  For languages where the orthography is not transparent, it may be helpful to include :code:`--num_pronunciations 3` so that more pronunciations are generated than just the most likely one. For more details on running G2P, see :ref:`g2p_dictionary_generating`.

Once you have looked over the dictionary, you can save the new pronunciations via:

.. code-block::

   mfa model add_words english_us_arpa ~/mfa_data/g2pped_oovs.txt

The new pronunciations will be available when you use  :code:`english_us_arpa` as the dictionary path in an MFA command, i.e. the modified command from :ref:`first_steps_align_pretrained`:

.. code-block::

   mfa align ~/mfa_data/my_corpus english_us_arpa english_us_arpa ~/mfa_data/my_corpus_aligned


.. warning::

   Please do look over the G2P results before adding them to the dictionary, at the very least to spot check.  Especially for non-transparent orthography systems, words with unseen graphemes, homographs, etc, G2P can generate phonotactically illegal forms, so I do not recommend piping G2P output to alignment without human spot checking.

.. _first_steps_align_train_acoustic_model:

Training a new acoustic model on a corpus
-----------------------------------------

For the purposes of this example, we'll also assume that you have done nothing else with MFA other than follow the :ref:`installation` instructions and you have the :code:`mfa` command working.  We'll assume that your :ref:`speech corpus <corpus_structure>` is stored in the folder :code:`~/mfa_data/my_corpus` and that you have a :ref:`pronunciation dictionary <dictionary_format>` at :code:`~/mfa_data/my_dictionary.txt`, so when working with your data, these paths will be the main thing to update.

The first thing we want to do is to make sure that the dataset is in the proper format for MFA, which is what the :code:`mfa validate` command does:

.. code-block::

   mfa validate ~/mfa_data/my_corpus ~/mfa_data/my_dictionary.txt

This command will look through the corpus and make sure that MFA is parsing everything correctly.  There are couple of different types of :ref:`corpus_structure` that MFA supports, but in general the core requirement is that you should have pairs of sound files and transcription files with the same name (except for the extension).  Take a look over the validator output and make sure that the number of speakers and number of files and utterances match your expectations, and that the number of Out of Vocabulary (OOV) items is not too high.  If you want to generate transcriptions for these words so that they can be aligned, see :ref:`first_steps_train_g2p` and :ref:`first_steps_g2p_pretrained` to make a new dictionary.  The validator will also attempt to run feature generation and train a simple monophone model to make sure that everything works within Kaldi.

Once we've validated the data, we can train an acoustic model (and output the aligned TextGrids if we want) it via the :code:`mfa train` command:

.. code-block::

   mfa train ~/mfa_data/my_corpus ~/mfa_data/my_dictionary.txt ~/mfa_data/new_acoustic_model.zip  # Export just the trained acoustic model
   mfa train ~/mfa_data/my_corpus ~/mfa_data/my_dictionary.txt ~/mfa_data/my_corpus_aligned  # Export just the training alignments
   mfa train ~/mfa_data/my_corpus ~/mfa_data/my_dictionary.txt ~/mfa_data/new_acoustic_model.zip --output_directory ~/mfa_data/my_corpus_aligned  # Export both trained model and alignments

As for other commands, if your data is large, you'll likely want to increase the number of jobs that MFA uses.  For that and more advanced configuration of the training command, see :ref:`train_acoustic_model`.

If training was successful, you'll now see the TextGrids in the output directory, assuming you wanted to export them. The TextGrid export is identical to if you had run :code:`mfa align` with the trained acoustic model.

If you choose export the acoustic model, you can now use this model for other utilities and use cases, such as refining your pronunciation dictionary through :ref:`training_dictionary` or :ref:`transcribing` for new data.  If you would like to store the exported acoustic model for easy reference like the downloaded pretrained models, you can save it via :code:`mfa model save`:

.. code-block::

   mfa model save acoustic ~/mfa_data/new_acoustic_model.zip

You can then run :code:`mfa model inspect` on it:

.. code-block::

   mfa model inspect acoustic new_acoustic_model

Or use it as a reference in other MFA commands.


.. _first_steps_train_g2p:

Training a G2P model from a pronunciation dictionary
----------------------------------------------------

For the purposes of this example, we'll also assume that you have done nothing else with MFA other than follow the :ref:`installation` instructions and you have the :code:`mfa` command working.  Finally, we'll assume that your pronunciation dictionary is stored as :code:`~/mfa_data/my_dictionary.txt` and that it fits the :ref:`dictionary_format`.


To train the G2P model, we use the :code:`mfa train_g2p`:

.. code-block::

   mfa train_g2p ~/mfa_data/my_dictionary.txt ~/mfa_data/my_g2p_model.zip

As for other commands, if your dictionary is large, you'll likely want to increase the number of jobs that MFA uses.  For that and more advanced configuration of the training command, see :ref:`g2p_model_training`.

Once the G2P model is trained, you should see the exported archive in the folder.  From here, we can save it for future use, or use the full path directly for generating pronunciations of new words.

.. code-block::

   mfa model save g2p ~/mfa_data/my_g2p_model.zip

   mfa g2p ~/mfa_data/my_new_word_list.txt my_g2p_model ~/mfa_data/my_new_dictionary.txt

   # Or

   mfa g2p ~/mfa_data/my_new_word_list.txt ~/mfa_data/my_g2p_model.zip ~/mfa_data/my_new_dictionary.txt

Take a look at :ref:`first_steps_g2p_pretrained` with this new model for a more detailed walk-through of generating a dictionary.

.. note::

   Please see :ref:`g2p_model_training_example` for an example using toy data.

.. _first_steps_tokenize:

Tokenize a corpus to add word boundaries
----------------------------------------

For the purposes of this example, we'll also assume that you have done nothing else with MFA other than follow the :ref:`installation` instructions and you have the :code:`mfa` command working.  Finally, we'll assume that your corpus is in Japanese and is stored in the folder :code:`~/mfa_data/my_corpus`, so when working with your data, this will be the main thing to update.

To tokenize the Japanese text to add spaces, first download the Japanese tokenizer model via:


.. code-block::

   mfa model download tokenizer japanese_mfa

Once you have the model downloaded, you can tokenize your corpus via:

.. code-block::

   mfa tokenize ~/mfa_data/my_corpus japanese_mfa ~/mfa_data/tokenized_version

You can check the tokenized text in :code:`~/mfa_data/tokenized_version`, verify that it looks good, and copy the files to replace the untokenized files in :code:`~/mfa_data/my_corpus` for use in alignment.

.. warning::

   MFA's tokenizer models are nowhere near state of the art, and I recommend using other tokenizers as they make sense:

   * Japanese: `nagisa <https://nagisa.readthedocs.io/en/latest/>`_
   * Chinese: `spacy-pkuseg <https://github.com/explosion/spacy-pkuseg/blob/master/readme/readme_english.md>`_
   * Thai: `sertiscorp/thai-word-segmentation <https://github.com/sertiscorp/thai-word-segmentation>`_

   The above were used in the initial construction of the training corpora for MFA, though the training segmentations for Japanese have begun to diverge from :code:`nagisa`, as they break up phonological words into morphological parses where for the purposes of acoustic model training and alignment it makes more sense to not split (nagisa: :ipa_inline:`使っ て [ts ɨ k a Q t e]` vs mfa: :ipa_inline:`使って [ts ɨ k a tː e]`). The MFA tokenizer models are provided as an easy start up path as the ones listed above may have extra dependencies and platform restrictions.

.. toctree::
   :maxdepth: 1
   :hidden:

   example
   tutorials
