
.. _transcribe_config:

*********************
Transcription options
*********************

.. csv-table::
   :widths: 20, 20, 60
   :header: "Parameter", "Default value", "Notes"

   "beam", 13, "Beam for decoding"
   "max_active", 7000, "Max active for decoding"
   "lattice_beam", 6, "Beam width for decoding lattices"
   "acoustic_scale", 0.083333, "Multiplier to scale acoustic costs"
   "silence_weight", 0.01, "Weight on silence in fMLLR estimation"
   "uses_speaker_adaptation", true, "Flag for whether to perform speaker adaptation"
   "first_beam", 10.0, "Beam for decoding in initial speaker-independent pass, only used if ``uses_speaker_adaptation`` is true"
   "first_max_active", 2000, "Max active for decoding in initial speaker-independent pass, only used if ``uses_speaker_adaptation`` is true"
   "fmllr_update_type", "full", "Type of fMLLR estimation"

Default transcriber config
--------------------------

.. code-block:: yaml

   beam: 13
   max_active: 7000
   lattice_beam: 6
   acoustic_scale: 0.083333
   silence_weight: 0.01
   fmllr: true
   first_beam: 10.0 # Beam used in initial, speaker-indep. pass
   first_max_active: 2000 # max-active used in initial pass.
   fmllr_update_type: full
