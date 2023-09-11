
.. _changelog_3.0:

*************
3.0 Changelog
*************

3.0.0a4
=======

- Separate out segmentation functionality into :ref:`create_segments` and :ref:`create_segments_vad`
- Fix a bug in :ref:`align_one` when specifying a ``config_path``

3.0.0a3
=======

- Refactor tokenization for future spacy use

3.0.0a2
=======

- Revamped how configuration is done following change to using threading instead of multiprocessing

3.0.0a1
=======

- Add dependency on :xref:`kalpy` for interacting for Kaldi
- Add command for :ref:`align_one`
- Migrate to threading instead of multiprocessing to avoid serializing Kalpy objects
