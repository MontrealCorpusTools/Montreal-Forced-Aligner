
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
   only Linux and MacOS. To use G2P functionality on Windows, please set up the "Windows Subsystem
   For Linux" and use the Bash console to continue the instructions.

   Also note that Kaldi binaries were built on fairly recent distributions (18.04 Ubuntu, 10.15 MacOSX Catalina).
   These binaries may not work on your specific distribution, particularly if is it older.  See :ref:`collect_binaries` for
   how to build and collect binaries on your machine for use with MFA.

1. Install Anaconda/Miniconda (https://docs.conda.io/en/latest/miniconda.html)
2. Create new environment via :code:`conda create -n aligner -c conda-forge openblas python=3.8 openfst pynini ngram baumwelch`

   a. On Windows to use the aligner natively without G2P functionality, use the command :code:`conda create -n aligner -c conda-forge openblas python=3.8`

3. Ensure you're in the new environment created (:code:`conda activate aligner`)
4. Run :code:`pip install montreal-forced-aligner`
5. Install third-party binaries via :code:`mfa thirdparty download` (see also :ref:`collect_binaries` to collect locally built binaries)
6. If you'd like to use :code:`sox` to align non-WAV files, install a system version (The conda package from :code:`conda install -c conda-forge sox` is
   missing a lot of codecs).

To upgrade to the latest version of MFA:

1. Activate your conda environment (i.e., :code:`conda activate aligner`)
2. Run :code:`pip install montreal-forced-aligner -U`

.. note::

   MFA 2.0.0a5 and earlier used Pynini version 2.1.0.  As of 2.0.0a6, versions have been upgraded to the latest version
   of Pynini, but there were some breaking changes, so please be sure to upgrade via :code:`conda upgrade -c conda-forge openfst pynini ngram baumwelch`
   if you installed a previous 2.0 alpha version to ensure correct performance.

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
(If this is not desired, delete or move the old model folder or use the ``--clean`` flag.)
You can specify your own temporary directory by using the ``-t``
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

.. warning::

   The prebuilt Kaldi binaries were built on Ubuntu 18.04 and MacOSX 10.15 (Catalina).  If you're using an older version
   of either of those, follow the instructions in :ref:`collect_binaries`.