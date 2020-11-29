
.. _`Montreal Forced Aligner releases`: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases

.. _`Kaldi GitHub repository`: https://github.com/kaldi-asr/kaldi

.. _`OpenGrm NGram library`: http://opengrm.org/NGramLibrary

.. _`Phonetisaurus`: https://github.com/AdolfVonKleist/Phonetisaurus

.. _installation:

************
Installation
************

All platforms
=============

1. Install Anaconda/Miniconda (https://docs.conda.io/en/latest/miniconda.html)
2. Create new environment via :code:`conda create -n aligner -c conda-forge python=3.8 openfst=1.7.6 pynini=2.1.0 ngram=1.3.9 baumwelch=0.3.1`
3. Run :code:`pip install montreal-forced-aligner`
4. Install third-party binaries:
   1. Download prebuilt ones via `mfa thirdparty download`
   2. Build and collect binaries for your specific platform via the instructions below


Building platform-specific binaries from scratch
================================================

1. Get kaldi compiled and working: `Kaldi GitHub repository`_
2. Collect the necessary binaries for MFA by running :code:`mfa thirdparty kaldi /path/to/kaldi/repo`

.. code-block:: bash

   mfa thirdparty validate


Files created when using the Montreal Forced Aligner
====================================================

The aligner will save data and logs for the models it trains in a new folder,
``Documents/MFA`` (which it creates in your user's home directory).  If a model for a corpus already
exists in MFA, it will use any existing models if you try to align it again.
(If this is not desired, delete or move the old model folder.)  You can specify your own temporary directory by using the ``-t``
flag when calling the executable.

