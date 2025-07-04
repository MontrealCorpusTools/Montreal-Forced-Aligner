.. _train_acoustic_model:

Train a new acoustic model ``(mfa train)``
******************************************

You can train new :term:`acoustic models` from scratch using MFA, and export the final alignments as :term:`TextGrids` at the end.  You don't need a ton of data to generate decent alignments (see `the blog post comparing alignments trained on various corpus sizes <https://memcauliffe.com/how-much-data-do-you-need-for-a-good-mfa-alignment.html>`_).  At the end of the day, it comes down to trial and error, so I would recommend trying different workflows of pretrained models vs training your own or adapting a model to your data to see what performs best.


.. note::

   You can use manual or verified reference alignments in training to bias the model to always use the reference phones
   when training acoustic models. Any utterance with manual alignments will always be included in training subsets. See :ref:`reference_alignment_format`
   for more information on how to include these alignments.


.. seealso::

   See the `train_acoustic_models.py <https://github.com/MontrealCorpusTools/mfa-models/blob/main/scripts/training_models/train_acoustic_models.py>`_
   in :xref:`mfa_model_scripts` for reference files and CLI commands that are used for training MFA pretrained models.

Phone topology
==============

The phone topology that MFA uses is different from the standard 3-state HMMs.  Each phone can have a maximum of 5 states, but allows for early exiting, so each phone has a minimum duration of 10ms (one MFCC frame) rather than 30ms for the 3-state HMM (three MFCC frames).

.. seealso::

   See :doc:`phone groups <../concepts/hmm>` for more information on HMMs and phone typologies.

   See `MFA models phone topologies for pretrained models <https://github.com/MontrealCorpusTools/mfa-models/tree/main/config/acoustic/topologies>`_
   in :xref:`mfa_models_repo` for examples.

Phone groups
============

By default each phone is treated independently of one another, which can lead to data sparsity issues or worse contextual modeling for clearly related phones when modeling triphones (i.e., long/short vowels :ipa_inline:`ɑ/ɑː`, stressed/unstressed versions :ipa_inline:`OY1/OY2/OY0`). Phone groups can be specified via the :code:`--phone_groups_path` flag. See :doc:`phone groups <../implementations/phone_groups>` for more information.

.. seealso::

   See `MFA models phone groups for pretrained models <https://github.com/MontrealCorpusTools/mfa-models/tree/main/config/acoustic/phone_groups>`_
   in :xref:`mfa_models_repo` for examples.

.. deprecated:: 3.0.0

   Using the :code:`--phone_set` flag to generate phone groups is deprecated as of MFA 3.0, please refer to using :code:`--phone_groups_path` flag to specify a phone groups configuration file instead.

Pronunciation modeling
======================

For the default configuration, pronunciation probabilities are estimated following the second and third SAT blocks.  See :ref:`training_dictionary` for more details.

A recent experimental feature for training acoustic models is the ``--train_g2p`` flag which changes the pronunciation probability estimation from a lexicon based estimation to instead using a G2P model as the lexicon. The idea here is that we have pronunciations generated by the initial blocks much like for the standard lexicon-based approach, but instead of estimating probabilities for individual word/pronunciation pairs and the likelihood of surrounding silence, it learns a mapping between the graphemes of the input texts and the phones.

.. seealso::

   See :doc:`phonological rules <../implementations/phonological_rules>` for how to specify regular expression-like phonological rules so you don't have to code every form for a regular rule.

   See `MFA models phonological rules for pretrained models <https://github.com/MontrealCorpusTools/mfa-models/tree/main/config/acoustic/rules>`_
   in :xref:`mfa_models_repo` for examples.

Language tokenization
=====================

By specifying a language via the :code:`--language` flag, tokenization will occur as part of text normalization.  This functionality is primarily useful for languages that do not rely on spaces to delimit words like Japanese, Thai, or Chinese languages.  If you're also using :code:`--g2p_model_path` to generate pronunciations during training, note that the language setting will require G2P models trained on specific orthographies (i.e., using :code:`mfa model download g2p korean_jamo_mfa` instead of :code:`mfa model download g2p korean_mfa`).


.. csv-table::
   :header: "Language", "Pronunciation orthography", "Input", "Output", "Dependencies", "G2P model"

   "Japanese", "Katakana", "これは日本語です", "コレ ワ ニホンゴ デス", ":xref:`sudachipy`", "`Katakana G2P <https://mfa-models.readthedocs.io/en/latest/g2p/Japanese/Japanese%20%28Katakana%29%20MFA%20G2P%20model%20v3_0_0.html>`_"
   "Korean", "Jamo", "이건 한국어야", "이건 한국어 야", ":xref:`python-mecab-ko`, :xref:`jamo`", "`Jamo G2P <https://mfa-models.readthedocs.io/en/latest/g2p/Korean/Korean%20%28Jamo%29%20MFA%20G2P%20model%20v3_0_0.html>`_"
   "Chinese", "Pinyin", "这是中文", "zhèshì zhōngwén", ":xref:`spacy-pkuseg`, :xref:`hanziconv`, :xref:`dragonmapper`", "`Pinyin G2P <https://mfa-models.readthedocs.io/en/latest/g2p/Mandarin/Mandarin%20%28China%20Pinyin%29%20MFA%20G2P%20model%20v3_0_0.html>`_"
   "Thai", "Thai script", "นี่คือภาษาไทย", "นี่ คือ ภาษาไทย", ":xref:`pythainlp`", "`Thai G2P <https://mfa-models.readthedocs.io/en/latest/g2p/Thai/Thai%20MFA%20G2P%20model%20v3_0_0.html>`_"

Command reference
=================


.. click:: montreal_forced_aligner.command_line.train_acoustic_model:train_acoustic_model_cli
   :prog: mfa train
   :nested: full

Configuration reference
=======================

.. seealso::

   See the yaml files in `MFA models acoustic configuration for pretrained models <https://github.com/MontrealCorpusTools/mfa-models/tree/main/config/acoustic>`_
   in :xref:`mfa_models_repo` for examples of using a file passed to ``--config_path``.

- :ref:`configuration_acoustic_modeling`

API reference
-------------

- :ref:`acoustic_modeling_api`

- :ref:`acoustic_model_training_api`
