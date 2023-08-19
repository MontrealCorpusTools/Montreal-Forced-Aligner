
.. _whats_new_3_0:

What's new in 3.0
=================

Version 3.0 of the Montreal Forced Aligner changes the way MFA interacts with Kaldi, shifting from calling binaries to using :xref:`kalpy` to provide access to Kaldi library functions and objects. This shift optimizes a number of processes, speeding up existing workflows and making new workflows like aligning single files without setting up a full corpus. See :ref:`changelog_3.0` for a more specific changes.

.. _3_0_kalpy:

Kalpy
-----

:xref:`kalpy` is a new library of Python bindings for the Kaldi libraries. In earlier versions of MFA, MFA relied on Kaldi executables to do everything involving Kaldi (i.e., compiling training graphs, aligning/transcribing files, computing LDA/fMLLR transforms), with MFA handling data preparation for corpora and lexicons.  What this meant was that MFA would call binaries like :code:`gmm-align-compiled` with command-line arguments pointing to objects stored on a hard disk like a file for the acoustic model, a file containing all the training graphs, or a file for features.  OS pipes were used, but they are limited to being able to pass single streams of information to a binary.  So if you wanted to pass an acoustic model, training graph, and features to :code:`gmm-align-compiled`, you would have to write at least two of those to the disk in order to be loaded again by :code:`gmm-align-compiled`.  Overall, less than ideal.

Kalpy allows for MFA to call Kaldi functions directly from Python, with arbitrary arguments that don't need to be serialized to the disk between MFA and Kaldi.  This opens up a lot more possibilities for optimization and working with intermediate Kaldi objects.

.. _3_0_align_one:

Aligning single files
---------------------

One key possibility with using Kalpy is to better handle smaller use cases. MFA has been optimized heavily for large corpora (particularly the ones used to generate the pretrained models).  That optimization does not work well for the use case of having a single file that needs to be aligned quickly, as the overhead from the setup dwarfs the actual processing time of the files, instead of vice versa.

MFA 3.0 includes a new command line utility for aligning a single file.  This utility doesn't use any database structures or speaker information, so it should be much faster than creating corpora of single files and trying to align them.  This command line utility is also light on creating temporary files, only caching the compiled lexicon FST. See :ref:`align_one` for more details.
