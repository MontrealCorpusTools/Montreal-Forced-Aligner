
.. _changelog_2.1:

*************
2.1 Changelog
*************

2.1.3
=====

- Fixed a bug with intervals after the end of the sound file having negative duration (they are now not parsed)
- Fixed an issue where utterances were not properly assigned to the correct channels
- Modified the logic for connections to attempt to solve error with too many clients

2.1.2
=====

- Fixed a crash in training when the debug flag was not set
- Set default postgresql port to 5433 to avoid conflicts with any system installations
- Fixed a crash in textgrid export

2.1.1
=====

- Fixed a bug with `mfa` command not working from the command line
- Updated to be compatible with PraatIO 6.0

2.1.0
=====

- Drop support for SQLite as a database backend
- Fixed a bug where TextGrid parsing errors would cause MFA to crash rather than ignore those files
- Updated CLI to use :xref:`click` rather than argparse
- Added :code:`--use_phone_model` flag for :code:`mfa align` and :code:`mfa validate` commands.  See :ref:`phone_models` for more details.
- Added :code:`--phone_confidence` flag for :code:`mfa validate` commands.  See :ref:`phone_models` for more details.
- Added modeling of :code:`cutoff` phones via :code:`--use_cutoff_model` which adds progressive truncations of the next word, if it's not unknown or a non-speech word (silence, laughter, etc). See :ref:`cutoff_modeling` for more details.
- Added support for using :xref:`speechbrain`'s VAD model in :ref:`create_segments`
- Overhaul and update :ref:`train_ivector`
- Overhaul and update :ref:`diarize_speakers`
- Added support for using :xref:`speechbrain`'s SpeakerRecognition model in :ref:`diarize_speakers`
