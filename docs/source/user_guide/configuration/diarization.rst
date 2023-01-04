
.. _configuration_diarization:

Diarization options
===================

.. csv-table::
    :widths: 20, 20, 60
    :header: "Parameter", "Default value", "Notes"
    :stub-columns: 1

    "cluster_type", ``optics``, "Clustering algorithm in :xref:`scikit-learn` to use, one of ``optics``, ``dbscan``, ``affinity``, ``agglomerative``, ``spectral, ``kmeans``"
    "expected_num_speakers", 0, "Number of speaker clusters to find, must be > 1 for ``agglomerative``, ``spectral``, and ``kmeans``"
    "sparse_threshold", 0.5, "Threshold on distance to limit precomputed sparse matrix"

.. _default_diarization_config:

Default diarization config file
-------------------------------

.. code-block:: yaml

    cluster_type: optics
    energy_mean_scale: 0.5
    max_segment_length: 30
    min_pause_duration: 0.05
