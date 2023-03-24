
.. _installation:

************
Installation
************

.. important::

   Kaldi and MFA are now built on :xref:`conda_forge` :fas:`party-horn`, so installation of third party binaries is wholly through conda from 2.0.0b4 onwards. Installing MFA via conda will pick up Kaldi as well.


General installation
====================

1. Install :xref:`miniconda`/:xref:`conda_installation`
2. Create new environment and install MFA: :code:`conda create -n aligner -c conda-forge montreal-forced-aligner`

   a.  You can enable the :code:`conda-forge` channel by default by running :code:`conda config --add channels conda-forge` in order to omit the :code:`-c conda-forge` from these commands

3. Ensure you're in the new environment created (:code:`conda activate aligner`)

Installing SpeechBrain
----------------------

1. Ensure you are in the conda environment created above
2. Install PyTorch
   a. CPU: :code:`conda install pytorch torchvision torchaudio cpuonly -c pytorch`
   b. GPU: :code:`conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`
3. Install Speechbrain via pip: :code:`pip install speechbrain`

.. _docker_installation:

Docker installation
===================

.. versionadded:: 2.2.6

   Docker images for MFA automatically built and available via `mmcauliffe/montreal-forced-aligner <https://hub.docker.com/repository/docker/mmcauliffe/montreal-forced-aligner>`_.


To use the Docker image of MFA:

1. Run :code:`docker image pull mmcauliffe/montreal-forced-aligner:latest`
2. Enter the interactive docker shell via :code:`docker run -it -v /path/to/data/directory:/data mmcauliffe/montreal-forced-aligner:latest`
3. Once you are in the shell, you can run MFA commands as normal (i.e., :code:`mfa align ...`).  You may need to download any pretrained models you want to use each session (i.e., :code:`mfa download acoustic english_mfa`)

.. important::

   For accessing system files, note the use of :code:`-v /path/to/data/directory:/data`, where the path before the colon is the local system path and the path after the colon is the mapped path inside docker.  For Windows, the path style is :code:`//c/Users/path`, note the slashes and how the drive is specified.

Installing MFA in your own containers
-------------------------------------

.. versionadded:: 2.2.6

   `Dockerfile for automatic releases <https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/blob/main/Dockerfile>`_


A simple Dockerfile for installing MFA would be:

.. code-block:: docker

   FROM condaforge/mambaforge:22.11.1-4 as build

   RUN mkdir -p /mfa
   RUN mamba create -p /env -c conda-forge montreal-forced-aligner

   RUN useradd -ms /bin/bash mfauser
   RUN chown -R mfauser /mfa
   RUN chown -R mfauser /env
   USER mfauser
   ENV MFA_ROOT_ENVIRONMENT_VARIABLE=/mfa
   RUN conda run -p /env mfa server init

   RUN echo "source activate /env && mfa server start" > ~/.bashrc
   ENV PATH /env/bin:$PATH

Crucially, note the useradd and subsequent user commands:

.. code-block:: docker

   RUN useradd -ms /bin/bash mfauser
   RUN chown -R mfauser /mfa
   RUN chown -R mfauser /env
   USER mfauser
   ENV MFA_ROOT_ENVIRONMENT_VARIABLE=/mfa
   RUN conda run -p /env mfa server init

These lines ensure that the database is initialized without using Docker's default root user, avoiding a permissions error thrown by PostGreSQL.

Upgrading from non-conda version
================================

In general, it's recommend to create a new environment.  If you want to update,

1. Activate your conda environment (i.e., :code:`conda activate aligner`)
2. Upgrade all packages via :code:`conda update --all`
3. Run :code:`pip uninstall montreal-forced-aligner` (to clean up previous pip installation)
4. Run :code:`conda install -c conda-forge montreal-forced-aligner`

.. _source_installation:

Installing from source
======================

If the Conda installation above does not work or the binaries don't work on your system, you can try building Kaldi and OpenFst from source, along with MFA.

1. Download/clone the :xref:`kaldi_github` and follow the installation instructions
2. If you're on Mac or Linux and want G2P functionality, install :xref:`openfst`, :xref:`opengrm_ngram`, :xref:`baumwelch`, and :xref:`pynini`
3. Make sure all Kaldi and other third party executables are on the system path
4. Download/clone the :xref:`mfa_github` and install MFA via :code:`python setup install` or :code:`pip install -e .`
5. Double check everything's working on the console with :code:`mfa -h`

.. note::

   You can also clone the conda-forge feedstocks for `OpenFst <https://github.com/conda-forge/openfst-feedstock>`_, `SoX <https://github.com/conda-forge/sox-feedstock>`_, `Kaldi <https://github.com/conda-forge/kaldi-feedstock>`_, and `MFA <https://github.com/conda-forge/montreal-forced-aligner-feedstock>`_ and run them with `conda build <https://docs.conda.io/projects/conda-build/en/latest/>`_ to build for your specific system.

Installing via pip
------------------

To install with pip and install minimal dependencies from conda:

1. Create a conda environment:

   * :fa:`fab fa-linux` Linux/:fa:`fab fa-apple` MacOSX: ``conda create -n aligner kaldi pynini``
   * :fa:`fab fa-windows` Windows: ``conda create -n aligner kaldi``

2. Activate environment via ``conda activate aligner``
3. Install MFA

   * From PyPi: ``pip install montreal-forced-aligner``
   * From :fa:`fab fa-github` GitHub: ``pip install git+https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner.git``
   * From inside the MFA repository root directory, you can install a local version via one of the following:

     * ``pip install -e .``
     * ``python setup.py install``
     * ``python setup.py develop``

MFA temporary files
===================

MFA uses a temporary directory for commands that can be specified in running commands with ``--temp_directory`` (or see :ref:`configuration`), and it also uses a directory to store global configuration settings and saved models.  By default this root directory is ``~/Documents/MFA``, but if you would like to put this somewhere else, you can set the environment variable ``MFA_ROOT_DIR`` to use that.  MFA will raise an error on load if it's unable to write the specified root directory.

Supported functionality
=======================

As of version 2.0.6, all features are available on all platforms.  Prior to this version, G2P and language model training was unavailable on native Windows, but could be used with Windows Subsystem for Linux (WSL).
