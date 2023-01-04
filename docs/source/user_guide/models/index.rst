.. _pretrained_models:

*****************
Pretrained models
*****************

The command for interacting with MFA models is :code:`mfa model`.  The subcommands allow for inspecting currently saved pretrained models, downloading ones from MFA's model repo, and saving models you have trained to be used with a simple name rather than the full path each time.

Following installation of MFA, :code:`mfa model list acoustic` will not list any models.  If you want to download the default English model trained on LibriSpeech, you can run :code:`mfa model download acoustic english_us_arpa`.  At which point, the previous ``list`` command will output "english_us_arpa" as an option.  When referring to an acoustic model in another MFA command, rather than the full path to the acoustic model, you can now supply just ``english_us_arpa`` and MFA will resolve it to the saved path.

Similarly, if you train a new model, you can run :code:`mfa model save acoustic /path/where/the/model/was/saved.zip`, then this model will be available via ``saved`` in the future.  The name defaults to whatever the archive is called without the directory or extension.  You can modify this name with the ``--name NEWNAME`` option

There are a number of pretrained models for aligning and generating pronunciation dictionaries. The command
for downloading these is :code:`mfa model download <model_type>` where ``model_type`` is one of ``acoustic``, ``g2p``, or
``dictionary``.

.. note::

   Please see the :xref:`mfa_models` site for information and statistics about various models.


Command reference
-----------------

.. click:: montreal_forced_aligner.command_line.model:model_cli
   :prog: mfa model
   :nested: full
