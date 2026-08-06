
.. _installation:

************
Installation
************


.. important::

   Montreal Forced Aligner is a command line utility.  For more information on how to get to the command line and use MFA, see :ref:`command_line`.


General installation
====================

Installing Conda
----------------

Conda is a command line utility for managing multiple software environments. Montreal Forced Aligner is distributed as a package that can be installed into a Conda environment, so the first step is to make sure the ``conda`` command is working. If you want to check if you have conda installed, you can run:

.. code-block:: console

   conda --help

If it prints out a long list of commands, then it's installed and you can go to installing/updating MFA.  If it reports that the conda command is not found, you can install the conda command by downloading an installer for your OS from `Miniforge downloads page <https://conda-forge.org/download/>`_, and running the installer via the steps at the bottom of the download page.

.. note::

   Miniforge also installs the command ``mamba`` which is a drop-in replacement command for ``conda``.  The key benefit of ``mamba`` over ``conda`` is that it's much faster when installing packages, but functionally will result in the same results.

Installing Montreal Forced Aligner
----------------------------------

Once ``conda`` is installed as above and can be invoked on the :ref:`command line <command_line>`, you can create a Conda environment with MFA installed via:

.. code-block:: console

   conda create -n aligner -c conda-forge montreal-forced-aligner -y


.. note::

   If you did not install conda via `Miniforge <https://conda-forge.org/download/>`_, then you may have to add the :code:`conda-forge` channel with the command:

   .. code-block:: console

      conda config --add channels conda-forge

Once the ``conda create`` command completes, you will have to activate the environment to get easy access to the ``mfa`` command:

.. code-block::

   conda activate aligner

From here you should be able to run MFA with the help flag to get a list of available MFA commands and their descriptions via:

.. code-block::

   mfa --help

If this is your first time installing MFA and you don't know exactly how to get started, see the :ref:`Alignment tutorial <alignment_example>` to align a demo corpus in English, Japanese, or Mandarin and get a sense of what MFA takes as inputs and produces as outputs.  If you want to start working with your own data and want to know the commands that MFA has to help get you to an aligned dataset, see the decision tree in :ref:`first_steps`.

Updating Montreal Forced Aligner
--------------------------------

To install the latest version, please run either :code:`conda update -c conda-forge montreal-forced-aligner kalpy kaldi=*=cpu* --update-deps` or  :code:`mamba update -c conda-forge montreal-forced-aligner kalpy kaldi=*=cpu* --update-deps` if you have mamba installed.

.. versionadded:: 3.0.5

   MFA version 3.0.5 onward has a helper utility for updating to new versions.  Run :code:`mfa_update` to fetch the latest versions of MFA, Kalpy, and Kaldi.

Installing SpeechBrain
----------------------

1. Ensure you are in the conda environment created above
2. Install PyTorch

   a. CPU: :code:`conda install pytorch torchvision torchaudio cpuonly -c pytorch`
   b. GPU: :code:`conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`

3. Install Speechbrain via pip: :code:`pip install speechbrain==1.0.3`

.. _older_installation:

Installing older versions of MFA
================================

If you need to use an older version of MFA, you can install it via:

.. code-block:: console

   conda install montreal-forced-aligner=X.X.X

More stable key versions:

* Stable 3.0 release: :code:`conda update -c conda-forge montreal-forced-aligner`
* Stable 2.2 release: :code:`conda install -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068`
* Stable 2.1 release: :code:`conda install -c conda-forge montreal-forced-aligner=2.1.7 openfst=1.8.2 kaldi=5.5.1068`
* Stable 2.0 release: :code:`conda install -c conda-forge montreal-forced-aligner=2.0.6 openfst=1.8.2 kaldi=5.5.1068`
* Stable 1.0 release: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/tag/v1.0.1

.. _docker_installation:

Docker installation
===================

.. versionadded:: 2.2.6

   Docker images for MFA automatically built and available via `mmcauliffe/montreal-forced-aligner <https://hub.docker.com/repository/docker/mmcauliffe/montreal-forced-aligner>`_.


To use the Docker image of MFA:

1. Run :code:`docker image pull mmcauliffe/montreal-forced-aligner:latest`
2. Enter the interactive docker shell via :code:`docker run -it -v /path/to/data/directory:/data mmcauliffe/montreal-forced-aligner:latest`
3. Once you are in the shell, you can run MFA commands as normal (i.e., :code:`mfa align ...`).  You may need to download any pretrained models you want to use each session (i.e., :code:`mfa model download acoustic english_mfa`)

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
   ENV MFA_ROOT_DIR=/mfa
   RUN conda run -p /env mfa server init

   RUN echo "source activate /env && mfa server start" > ~/.bashrc
   ENV PATH /env/bin:$PATH

Crucially, note the useradd and subsequent user commands:

.. code-block:: docker

   RUN useradd -ms /bin/bash mfauser
   RUN chown -R mfauser /mfa
   RUN chown -R mfauser /env
   USER mfauser
   ENV MFA_ROOT_DIR=/mfa
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
