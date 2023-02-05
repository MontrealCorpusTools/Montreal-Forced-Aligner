
.. _changelog_2.1:

*************
2.1 Changelog
*************

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
