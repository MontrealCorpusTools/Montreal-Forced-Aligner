
.. _server:

***********
MFA Servers
***********

MFA database servers
====================

By default, MFA starts or creates a PostgreSQL servers when a command is invoked, and stops the server at the end of processing.  The goal here is to have as unobtrusive of a database server as possible, however there are use cases that you may require more control. To turn off the automatic management of PostgreSQL servers, run :code:`mfa configure --disable_auto_server`.

You can have multiple PostgreSQL servers by using the :code:`--profile` flag, if necessary.  By default the "global" profile is used.  The profile flags are used in :ref:`configure_cli`, as the default options set with :code:`configure` are done on a per-profile basis.


PostgreSQL configuration
------------------------

MFA overrides some default configuration values for its PostgreSQL servers when they are initialized.

.. code-block::

   log_min_duration_statement = 5000
   enable_partitionwise_join = on
   enable_partitionwise_aggregate = on
   unix_socket_directories = '/path/to/current/profile/socket_directory'
   listen_addresses = ''

   maintenance_work_mem = 1GB
   work_mem = 128MB
   shared_buffers = 256MB
   max_connections = 1000

The goal for MFA is to run on local desktops at reasonable performance on moderate sized corpora (<3k hours).  Depending on your use case, you may need to tune the :code:`postgres.conf` file further to suit your set up and corpus (see `PostgreSQL's documentation <https://www.postgresql.org/docs/15/runtime-config.html>`_ and `postgresqltuner utility script <https://github.com/jfcoz/postgresqltuner>`_.  Additionally, note that any port listening is turned off by default and connections are handled via socket directories.

.. warning::

   MFA PostgreSQL databases are meant to be on the expendable side. Though they can persist across use cases, it's not really recommended.  Use of :code:`--clean` drops all data in the database to ensure a fresh start state, as various commands perform destructive commands.  As an example :ref:`create_segments` deletes and recreates :class:`~montreal_forced_aligner.db.Utterance` objects, so the original text transcripts are absent in the database following its run.

.. _server_cli:

Managing MFA database servers
=============================

MFA PostgreSQL servers can be managed via the subcommands in `mfa server`, allowing you to initialize new servers, and start, stop, and delete existing servers.

.. click:: montreal_forced_aligner.command_line.server:server_cli
   :prog: mfa server
   :nested: full

API reference
-------------

- :ref:`server_api`
