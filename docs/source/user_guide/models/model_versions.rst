.. _mfa_model_versions:

******************
MFA model versions
******************

MFA models have historically been compressed zip archives with distinct purposes, i.e. separation of acoustic models from pronunciation dictionaries and G2P models.  As of 3.4 a new model format is being introduced to better integrate with distribution via :xref:`hf`.

The primary difference with the new HF model format is that models are no longer zip archives, but directories that can be uploaded and tracked in version control.  Additionally, a single HF model contains everything need to perform alignment: acoustic model, pronunciation dictionaries, and G2P models.

Command line signatures
=======================

Historically for :ref:`pretrained_alignment`, the command line signature has been:

.. code-block:: bash

   mfa align CORPUS DICTIONARY ACOUSTIC_MODEL OUTPUT

Because the HF models contain dictionaries and acoustic models, the new signature for :ref:`pretrained_alignment_hf` will be:


.. code-block:: bash

   mfa align_hf CORPUS HF_MODEL OUTPUT

Where ``HF_MODEL`` will correspond to any organization's model on Hugging Face.  For instance, the full specification for MFA's English model is ``MontrealCorpusTools/english_mfa``, which when used will download the model from https://huggingface.co/MontrealCorpusTools/english_mfa automatically the first time it's used.

.. note::

   The HF model string can use a commit or version specified with ``@``, such as ``MontrealCorpusTools/english_mfa@3.3.0`` or ``english_mfa@04a2961``.  Additionally, if only the model name is specified and not an organization's name, then something like ``english_mfa`` will be interpreted as MFA's ``MontrealCorpusTools/english_mfa``.

For models with multiple dialects/dictionaries, such as ``english_mfa``, the dictionary to use can be specified with the ``--dialect`` flag, i.e. ``--dialect english_india`` to use the Indian English dictionary.

To use G2P models to generate pronunciations for words not in the dictionary, the flag ``--use_g2p`` can be used, instead of the legacy way of specifying the path to a G2P model via ``--g2p_model_path``.


The same changes in signature for ``mfa adapt`` for adding a ``mfa adapt_hf`` command:

.. code-block:: bash

   mfa adapt_hf CORPUS HF_MODEL ADAPTED_HF_MODEL --dialect DIALECT --use_g2p

The ADAPTED_HF_MODEL argument is now a directory instead of a zip file.  For training, there is no command in signature, but supplying a directory instead of a zip file will lead to training a HF model:

.. code-block:: bash

   mfa train CORPUS DICTIONARY OUTPUT_DIRECTORY

   # or legacy acoustic models:

   mfa train CORPUS DICTIONARY OUTPUT_ZIP_FILE

If the output path is a directory (i.e., no ``.zip`` ending), then the training procedure will include an additional training of G2P models for each dictionary.  If a G2P model is specified via ``--g2p_model_path`` for the training command, then the G2P model will simply be exported to the necessary path in the HF model.

.. note::

   Using a pretrained G2P model is helpful for cases where the default orthography is not being used for generating the pronunciations.  For instance, for Japanese, the tokenizer provides pronunciations in katakana, so the pretrained `MontrealCorpusTools/japanese_mfa <https://huggingface.co/MontrealCorpusTools/japanese_mfa>`_ model has a G2P model that is just based off of katakana.  Likewise for Mandarin, the characters are converted to Pinyin before generating pronunciations, so the relevant G2P models in `MontrealCorpusTools/mandarin_mfa <https://huggingface.co/MontrealCorpusTools/mandarin_mfa>`_ are based off of Pinyin orthography rather than Hanzi.

.. _hf_model_details:

MFA model structure
===================

The file structure of MFA models for distribution via Hugging Face is::

    +-- english_mfa
    |   --- acoustic
    |   --- dictionary
    |   --- g2p
    |   --- LICENSE
    |   --- README.md


The ``acoustic`` directory contains all the information and files related to the HMM-GMM modeling.  This file directory contents are identical to zip archives that were created via the ``mfa train`` command in previous versions.

The ``dictionary`` directory contains all dictionaries (text files with ``.dict`` extension) that were trained with the acoustic model, along with a ``rules.yaml`` file that contains information about the probability of phonological rule application if these were supplied to the training command.

The ``g2p`` directory contains subdirectories for each dialect/dictionary that was used in training.  The names of the folders will correspond to the names of the dictionaries in the ``dictionary`` directory.

The ``LICENSE`` file contains the model's license, by default MFA generates a CC BY-4.0 license, but this is configurable. The model card (``README.md``) contains information about the training data for the model, specifically the phone set of the dictionaries/G2P models, the number of hours and any metadata about various corpora that made up the training data.

.. _model_distribution:

Distributing pretrained models
==============================

Training a distributable model
------------------------------

As part of ``mfa train`` when specifying a HF-compatible model as the output, a basic CC BY-4.0 ``LICENSE`` file and ``README.md`` file will be generated. The ``README.md`` model card are intended to largely follow `Hugging Face's recommendations for model cards <https://huggingface.co/docs/hub/en/model-cards>`_, but some recommendations are not relevant for MFA in particular.

These files can be edited and updated after the training is complete to remove placeholders, or a JSON file with metadata about the model can be provided to the ``mfa train`` command:

.. code-block:: bash

   mfa train CORPUS_DIRECTORY DICTIONARY_PATH OUTPUT_MODEL_DIRECTORY --metadata_path /path/to/metadata.json


The metadata JSON can take the following fields:

.. csv-table::
   :header: "JSON field", "Description", "Default value"
   :widths: 50, 150, 150

   "model_description", "Description of the model, language variety it was trained on.", ""
   "developers", "Developers of the model", "[More Information Needed]"
   "funded_by", "Any funding sources to recognize", "[More Information Needed]"
   "language", "The language(s) for this model", "[More Information Needed]"
   "language_code", "The language code(s) for this model", "und (for `Undetermined <https://en.wikipedia.org/wiki/ISO_639-3#Non-language_codes>`_)"
   "license", "The license for distributing the model", "cc-by-4.0"
   "preprocessing", "Any preprocessing steps done with the data", "[More Information Needed]"
   "citation_bibtex", "Citation for the model in Bibtex format", "[More Information Needed]"
   "citation_apa", "Citation for the model in APA format", "[More Information Needed]"
   "contact_details", "Contact details for questions about the model", "[More Information Needed]"
   "direct_use", "The direct use of the model, including what language varieties and styles that it should align well", "This model is intended to be used for forced alignment of speech varieties that it was trained on."
   "out_of_scope_use", "What tasks can the model be applied to that are out of scope and so will likely provide unreliable results", "This model cannot provide accurate assessments of goodness of pronunciations or provide transcripts."
   "bias_risks_limitations", "Details on the bias in the training data, risks and limitations in applying this model to new data", "This model will perform best on the variety of speech that it was trained on (dialect/language/demographics)."
   "bias_recommendations", "Any recommendations for mitigating bias in the model from its training data", "When using this model on a variety that it was not trained on, better results can be attained by adapting the model to the data to be aligned first."
   "get_started_code", "Any code blocks/installation instructions for using the model", "To get started, follow the instructions for `installing MFA <https://montreal-forced-aligner.readthedocs.io/en/latest/getting_started.html>`_."
   "software", "Any software that used for processing data or training the model", "This model was trained via the `Montreal Forced Aligner <https://montreal-forced-aligner.readthedocs.io/>`_."

.. seealso::

   The :xref:`mfa_models_repo` has the metadata JSON files used to train MFA models `here <https://github.com/MontrealCorpusTools/mfa-models/tree/main/config/acoustic/metadata>`_.

Converting a legacy model to be compatible with Hugging Face
------------------------------------------------------------

MFA 3.4 adds a helper command to migrate legacy acoustic model and dictionaries (and G2P models) to the new model format:

.. code-block:: bash

   mfa model create_hf_model ACOUSTIC_MODEL_PATH DICTIONARY_PATH OUTPUT_MODEL_PATH --g2p_model_path G2P_MODEL_PATH

This command will create a new directory that you can upload with the acoustic model folder, the dictionary folder, and a G2P model specified.  If a G2P model is not specified, one will be trained (you can pass a config file via ``--config_path`` or keyword arguments the same as for :ref:`g2p_model_training`.  You can pass a metadata JSON file as above that will be used to fill out the generated ``README.md`` file, which will do some inspection of the acoustic model for model training details (though these are much reduced from the details available when training from scratch) and the dictionary for filling out an IPA chart.


Uploading the model to Hugging Face
-----------------------------------

Once you are happy with the license and the model card, you can upload the model via:

.. code-block:: bash

   mfa model upload MODEL_PATH REPO_ID

Where REPO_ID is a organization/user's repo id like ``MontrealCorpusTools/english_mfa``.  In order for the upload to be successful, you need to have an account with write permissions, and `ensure you're authenticated with Hugging Face on the command line <https://huggingface.co/docs/huggingface_hub/quick-start#authentication>`_.

If the repository specified does not exist, MFA will attempt to create it, provided you've authenticated with an account that has the correct permissions.  Then MFA will sync all the files in the model's local directory to the Hugging Face repo, if there are changes, then a new commit will be created.  If you'd like to tag this commit with a version string, you can do so via the ``--version`` string.  If this version is already a tag for a previous commit, by default MFA will not update the version to the latest commit, so for convenience, you can specify ``--overwrite_version`` to ensure that the latest commit has the specified version.
