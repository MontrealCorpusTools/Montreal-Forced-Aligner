
***************
Getting started
***************


.. important::

   Montreal Forced Aligner is a command line utility.  For more information on how to get to the command line and use MFA, see :ref:`command_line`.

Installation
------------

.. grid:: 2

    .. grid-item-card:: Installing with conda
       :text-align: center
       :columns: 12

       MFA can be easily installed once conda is available via `Miniforge <https://conda-forge.org/download/>`_ or another source:

       .. code-block:: console

          conda config --add channels conda-forge
          conda create -n aligner montreal-forced-aligner -y
          conda activate aligner
          mfa --help

       +++

       .. button-link:: https://conda-forge.org/download/
          :color: primary
          :expand:

          Install Miniforge


    .. grid-item-card:: In-depth instructions
       :text-align: center

       Using :ref:`Docker <docker_installation>`? Want to :ref:`install via source <source_installation>`? Want a step-by-step instructions for installing MFA?

       +++

       .. button-ref:: installation
          :expand:
          :color: primary
          :ref-type: doc

          To the installation guide


    .. grid-item-card:: First steps
       :text-align: center

       Once MFA is installed, try it out aligning a demo corpus in English, Japanese, or Mandarin and make sure everything's working and what the inputs/outputs of MFA are

       +++

       .. button-ref:: alignment_example
          :expand:
          :color: primary

          Align a demo corpus


.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   first_steps/index
   first_steps/alignment_example
   first_steps/remapping_example
   first_steps/tutorials
