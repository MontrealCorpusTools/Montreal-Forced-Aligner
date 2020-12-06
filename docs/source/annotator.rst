.. _annotator:

*********
Annotator
*********

.. attention::

   The GUI annotator is under development and is currently pre-alpha. Use at your own risk and please use version control
   or back up any critical data.

Currently the functionality of the Annotator GUI allows for users to modify transcripts and add/change
entries in the pronunciation dictionary to interactively fix out of vocabulary issues.

.. warning::

   If you are trying to use the annotator from Windows, extra steps are required.  Within the bash
   subsystem, install the Qt5 prerequsites as follows:

  .. code-block:: bash

     sudo apt-get install qt5-default libqt5multimedia5-plugins libxcb-xinerama0

To use the annotator, first follow the instructions in :ref:`installation`.  Once MFA is installed and thirdparty binaries
have been downloaded, run the following command:

.. code-block:: bash

    mfa annotator

Initial setup
=============

To load a corpus for inspection, go to the Corpus drop down menu and select "Load a corpus".  Navigate
to the desired corpus directory.  Please note that it should follow one of the data formats outlined in :ref:`data_format`.

Next, dictionary files and G2P models should be loaded via their respective menus.  If any pretrained
models have been installed via :ref:`pretrained_models`, these can be selected directly.

Fixing out of vocabulary issues
===============================

Once the corpus is loaded with a dictionary, utterances in the corpus will be parsed for whether they contain
an out of vocabulary (OOV) word.  If they do, they will be marked in that column on the left with a red cell.

To fix a transcript, click on the utterance in the table.  This will bring up a detail view of the utterance,
with a waveform window above and the transcript in the text field.  Clicking the ``Play`` button will allow you
to listen to the audio.  The utterance text can be freely edited, any changes can be reverted
via the ``Reset`` button.  Pressing the ``Save`` button will save the utterance text to the .lab/.txt file
or update the interval in the TextGrid.

.. warning::

   Clicking ``Save`` will overwrite the source file loaded, so use this software with caution.
   Backing up your data and/or using version control is recommended to ensure that any data loss
   during corpus creation is minimized.

If the word causing the OOV warning is in fact a word you would like aligned, you can right click on
the word and select ``Add pronunciation for 'X'`` if a G2P model is loaded.  This will run the G2P
model to generate a pronunciation in the dictionary which can then be modified if necessary and the dictionary
can be saved via the ``Save dictionary`` button.  You can also look up any word in the pronunciation
dictionary by right clicking and selecting ``Look up 'X' in dictionary``.  Any pronunciation can be modified
and saved.  The ``Reset dictionay`` button wil discard any changes made to the dictionary.
