
.. _workflows_index:

Workflows available
===================

The primary workflow in MFA is forced alignment, where text is aligned to speech along with phones derived from a pronunciation dictionary and an acoustic model. There are, however, other workflows for transcribing speech using speech-to-text functionality in Kaldi, pronunciation dictionary creation using Pynini, and some basic corpus creation utilities like VAD-based segmentation. Additionally, acoustic models, G2P models, and language models can be trained from your own data (and then used in alignment and other workflows).

.. warning::

   Speech-to-text functionality is pretty basic, and the model architecture used in MFA is older GMM-HMM and NGram models, so using something like :xref:`coqui` or Kaldi's ``nnet`` functionality will likely yield better quality transcriptions.

.. hint::

   See :ref:`pretrained_models` for details about commands to inspect, download, and save various pretrained MFA models.

.. toctree::
   :hidden:

   alignment
   adapt_acoustic_model
   train_acoustic_model
   dictionary_generating
   g2p_train
