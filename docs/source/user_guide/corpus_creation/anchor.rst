
.. _`Anchor Annotator documentation`: https://anchor-annotator.readthedocs.io/en/latest/

.. _anchor:

Anchor annotator ``(mfa anchor)``
=================================

The Anchor Annotator is a GUI utility for MFA that allows for users to modify transcripts and add/change entries in the pronunciation dictionary to interactively fix out of vocabulary issues.

.. attention::

   Anchor is under development and is currently pre-alpha. Use at your own risk and please use version control or back up any critical data.


To use the annotator, first install the anchor subpackage:

.. code-block::

   conda install montreal-forced-aligner[anchor]

This will install MFA if hasn't been along with all the packages that Anchor requires.  Once installed, Anchor can be started with the following MFA subcommand `mfa anchor`.

See the `Anchor Annotator documentation`_ for more information.

Command reference
=================


.. click:: montreal_forced_aligner.command_line.anchor:anchor_cli
   :prog: mfa anchor
   :nested: full
