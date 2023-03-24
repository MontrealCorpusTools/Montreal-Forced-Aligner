
***************
Getting started
***************


Installation
------------

.. grid:: 2

    .. grid-item-card:: Installing with conda
       :text-align: center
       :columns: 12

       MFA is now on :xref:`conda_forge` and can be installed with Anaconda or Miniconda:

       .. code-block:: bash

          conda config --add channels conda-forge
          conda install montreal-forced-aligner

       +++

       .. button-link:: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
          :color: primary
          :expand:

          Install Conda


    .. grid-item-card:: In-depth instructions
       :text-align: center

       Using :ref:`Docker <docker_installation>`? Want to :ref:`install via source <source_installation>`?

       +++

       .. button-ref:: installation
          :expand:
          :color: primary
          :ref-type: doc

          To the installation guide


    .. grid-item-card:: First steps
       :text-align: center

       First time using MFA? Want a walk-through of a specific use case?

       +++

       .. button-ref:: first_steps
          :expand:
          :color: primary

          First steps


.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   first_steps/index
