
.. _configuration:

*************
Configuration
*************

MFA root directory
==================

MFA uses a temporary directory for commands that can be specified in running commands with ``--temp_directory`` (see below), and it also uses a directory to store global configuration settings and saved models.  By default this root directory is ``~/Documents/MFA``, but if you would like to put this somewhere else, you can set the environment variable ``MFA_ROOT_DIR`` to use that.  MFA will raise an error on load if it's unable to write to the root directory.

.. _configure_cli:

Global configuration
====================

Global configuration for MFA can be updated via the ``mfa configure`` subcommand. Once the command is called with a flag, it will set a default value for any future runs (though, you can overwrite most settings when you call other commands).

.. click:: montreal_forced_aligner.command_line.configure:configure_cli
   :prog: mfa configure
   :nested: full

Configuring specific commands
=============================

MFA has the ability to customize various parameters that control aspects of data processing and workflows.  These can be supplied via the command line like:

.. code-block:: bash

   mfa align ... --beam 1000

The above command will set the beam width used in aligning to ``1000`` (and the retry beam width to 4000).  This command is the equivalent of supplying a config file like the below via the ``--config_path``:

.. code-block:: yaml

   beam: 1000

Supplying the above via:

.. code-block:: bash

   mfa align ... --config_path config_above.yaml

will also set the beam width to ``1000`` and retry beam width to ``4000`` as well.

For simple settings, the command line argument approach can be good, but for more complex settings, the config yaml approach will allow you to specify things like aspects of training blocks or punctuation:

.. code-block:: yaml

   beam: 100
   retry_beam: 400

   punctuation: ":,."

   training:
     - monophone:
         num_iterations: 20
         max_gaussians: 500
         subset: 1000
         boost_silence: 1.25

     - triphone:
         num_iterations: 35
         num_leaves: 2000
         max_gaussians: 10000
         cluster_threshold: -1
         subset: 5000
         boost_silence: 1.25
         power: 0.25

You can then also override these options on the command like, i.e. ``--beam 10 --config_path config_above.yaml`` would reset the beam width to ``10``.  Command line specified arguments always have higher priority over the parameters derived from a configuration yaml.

.. toctree::
   :hidden:

   global.rst
   acoustic_modeling.rst
   acoustic_model_adapt.rst
   g2p.rst
   lm.rst
   transcription.rst
   segment.rst
   ivector.rst
   diarization
