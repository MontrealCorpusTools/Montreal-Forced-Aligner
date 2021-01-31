
.. _`Montreal Forced Aligner releases`: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases

.. _`Kaldi GitHub repository`: https://github.com/kaldi-asr/kaldi

.. _installation:

************
Installation
************

All platforms
=============

.. warning::

   Windows native install is not fully supported in 2.0.  G2P functionality will be unavailable due to Pynini supporting
   only Linux and MacOS. To use the aligner properly on Windows, please set up the "Windows Subsystem
   For Linux" and use the Bash console to continue the instructions.

1. Install Anaconda/Miniconda (https://docs.conda.io/en/latest/miniconda.html)
2. Create new environment via :code:`conda create -n aligner -c conda-forge openblas python=3.8 openfst=1.7.6 pynini=2.1.0 ngram=1.3.9 baumwelch=0.3.1`
3. Run :code:`pip install montreal-forced-aligner`
4. Install third-party binaries via :code:`mfa thirdparty download` (see also :ref:`collect_binaries` to collect locally built binaries)

.. _collect_binaries:

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

Supported functionality
=======================

Currently in the 2.0 alpha, supported functionality is somewhat fragmented across platforms.  Native support for features
is as follows.  Note that Windows can use Windows Subsystem for Linux to use the Linux version as necessary.

.. csv-table::
   :header: "Feature", "Linux support", "Windows support", "MacOS support"

   "Alignment", "Yes", "Yes", "Yes"
   "G2P", "Yes", "No", "Yes"
   "Transcribe", "Yes", "Yes", "Yes"
   "Train LM", "Yes", "No", "Yes"
   "Train dictionary", "Yes", "Yes", "Yes"
