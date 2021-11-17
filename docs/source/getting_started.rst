
.. _`Conda forge`: https://conda-forge.org/

.. _getting_started_ref:

***************
Getting started
***************


Installation
------------

.. panels::
    :card: + install-card
    :column: col-lg-6 col-md-6 col-sm-12 col-xs-12 p-3

    Installing with conda
    ^^^^^^^^^^^^^^^^^^^^^

    MFA is now on `Conda forge`_
    and can be installed with Anaconda or Miniconda:

    +++

    .. code-block:: bash

        conda config --add channels conda-forge
        conda install montreal-forced-aligner

    +++

    .. link-button:: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
            :type: url
            :text: Install Conda
            :classes: btn-block btn-primary btn-navigation stretched-link


    ---

    In-depth instructions
    ^^^^^^^^^^^^^^^^^^^^^

    Want to learn more about installing? Want to use G2P commands on Windows?

    +++

    .. link-button:: installation_ref
            :type: ref
            :text: To the installation guide
            :classes: btn-block btn-primary btn-navigation stretched-link

    ---
    :column: col-12 p-3

    First steps
    ^^^^^^^^^^^

    First time using MFA? Want a walkthrough of a specific use case?


    .. link-button:: first_steps
            :type: ref
            :text: First steps
            :classes: btn-block btn-primary btn-navigation


.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   first_steps/index
