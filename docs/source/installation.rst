
.. _installation_ref:

************
Installation
************

.. important::

   Kaldi and MFA are now built on :xref:`conda_forge` |:tada:|, so installation of third party binaries is wholly through conda from 2.0.0b4 onwards. Installing MFA via conda will pick up Kaldi as well.


All platforms
=============

1. Install Anaconda/Miniconda (https://docs.conda.io/en/latest/miniconda.html)
2. Create new environment and install MFA: :code:`conda create -n aligner -c conda-forge montreal-forced-aligner`

   a.  You can enable the :code:`conda-forge` channel by default by running :code:`conda config --add channels conda-forge` in order to omit the :code:`-c conda-forge` from these commands

3. Ensure you're in the new environment created (:code:`conda activate aligner`)

Upgrading from non-conda version
================================

In general, it's recommend to create a new environment.  If you want to update,

1. Activate your conda environment (i.e., :code:`conda activate aligner`)
2. Upgrade all packages via :code:`conda update --all`
3. Run :code:`pip uninstall montreal-forced-aligner` (to clean up previous pip installation)
4. Run :code:`conda install -c conda-forge montreal-forced-aligner`

.. warning::

   Windows native install is not fully supported in 2.0.  G2P functionality will be unavailable due to Pynini supporting only Linux and MacOS. To use G2P functionality on Windows, please set up the :xref:`wsl` and use the Bash console to continue the instructions.

Supported functionality
=======================

Currently in the 2.0 beta, supported functionality is fragmented across platforms.  Native support for features
is as follows.  Note that Windows can use Windows Subsystem for Linux to use the Linux version as necessary.

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - Feature
     - Linux support
     - Windows support
     - MacOS support

   * - Alignment
     - .. raw:: html

          <span class='rst-table-cell supported'>Yes</span>
     - .. raw:: html

          <span class='rst-table-cell supported'>Yes</span>
     - .. raw:: html

          <span class='rst-table-cell supported'>Yes</span>

   * - G2P training
     - .. raw:: html

          <span class='rst-table-cell supported'>Yes</span>
     - .. raw:: html

          <span class='rst-table-cell not-supported'>No</span>
     - .. raw:: html

          <span class='rst-table-cell supported'>Yes</span>

   * - G2P generation
     - .. raw:: html

          <span class='rst-table-cell supported'>Yes</span>
     - .. raw:: html

          <span class='rst-table-cell not-supported'>No</span>
     - .. raw:: html

          <span class='rst-table-cell supported'>Yes</span>

   * - Transcription
     - .. raw:: html

          <span class='rst-table-cell supported'>Yes</span>
     - .. raw:: html

          <span class='rst-table-cell supported'>Yes</span>
     - .. raw:: html

          <span class='rst-table-cell supported'>Yes</span>

   * - Training language model
     - .. raw:: html

          <span class='rst-table-cell supported'>Yes</span>
     - .. raw:: html

          <span class='rst-table-cell not-supported'>No</span>
     - .. raw:: html

          <span class='rst-table-cell supported'>Yes</span>
