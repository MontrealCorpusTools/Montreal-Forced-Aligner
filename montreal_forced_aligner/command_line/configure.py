import rich_click as click

from montreal_forced_aligner import config

__all__ = ["configure_cli"]


@click.command(
    "configure",
    help="The configure command is used to set global defaults for MFA so "
    "you don't have to set them every time you call an MFA command.",
)
@click.option(
    "-p",
    "--profile",
    "profile",
    help='Configuration profile to use, defaults to "global"',
    type=str,
    default=None,
)
@click.option(
    "--temporary_directory",
    "-t",
    help=f"Set the default temporary directory."
    f"Currently defaults to {config.TEMPORARY_DIRECTORY}",
    type=str,
    default=None,
)
@click.option(
    "--num_jobs",
    "-j",
    help=f"Set the number of processes to use by default. "
    f"Currently defaults to {config.NUM_JOBS}",
    type=int,
    default=None,
)
@click.option(
    "--always_clean/--never_clean",
    "clean",
    help="Turn on/off clean mode where MFA will clean temporary files before each run. "
    f"Currently defaults to {config.CLEAN}.",
    default=None,
)
@click.option(
    "--always_verbose/--never_verbose",
    "verbose",
    help="Turn on/off verbose mode where MFA will print more output. "
    f"Currently defaults to {config.VERBOSE}.",
    default=None,
)
@click.option(
    "--always_quiet/--never_quiet",
    "quiet",
    help="Turn on/off quiet mode where MFA will not print any output. "
    f"Currently defaults to {config.QUIET}.",
    default=None,
)
@click.option(
    "--always_debug/--never_debug",
    "debug",
    help="Turn on/off extra debugging functionality. " f"Currently defaults to {config.DEBUG}.",
    default=None,
)
@click.option(
    "--always_overwrite/--never_overwrite",
    "overwrite",
    help="Turn on/off overwriting export files. " f"Currently defaults to {config.OVERWRITE}.",
    default=None,
)
@click.option(
    "--enable_mp/--disable_mp",
    "use_mp",
    help="Turn on/off multiprocessing. "
    "Multiprocessing is recommended will allow for faster executions. "
    f"Currently defaults to {config.USE_MP}.",
    default=None,
)
@click.option(
    "--enable_textgrid_cleanup/--disable_textgrid_cleanup",
    "cleanup_textgrids",
    help="Turn on/off post-processing of TextGrids that cleans up "
    "silences and recombines compound words and clitics. "
    f"Currently defaults to {config.CLEANUP_TEXTGRIDS}.",
    default=None,
)
@click.option(
    "--enable_auto_server/--disable_auto_server",
    "auto_server",
    help="If auto_server is enabled, MFA will start a server at the beginning of a command and close it at the end. "
    "If turned off, use the `mfa server` commands to initialize, start, and stop a profile's server. "
    f"Currently defaults to {config.AUTO_SERVER}.",
    default=None,
)
@click.option(
    "--enable_use_postgres/--disable_use_postgres",
    "use_postgres",
    help="If use_postgres is enabled, MFA will use PostgreSQL as the database backend instead of sqlite. "
    f"Currently defaults to {config.USE_POSTGRES}.",
    default=None,
)
@click.option(
    "--blas_num_threads",
    help="Number of threads to use for BLAS libraries, 1 is recommended "
    "due to how much MFA relies on multiprocessing. "
    f"Currently defaults to {config.BLAS_NUM_THREADS}.",
    type=int,
    default=None,
)
@click.option(
    "--github_token",
    default=None,
    help="Github token to use for model downloading.",
    type=str,
)
@click.option(
    "--bytes_limit",
    default=None,
    help="Bytes limit for Joblib Memory caching on disk.",
    type=int,
)
@click.option(
    "--seed",
    default=None,
    help="Random seed to set for various pseudorandom processes.",
    type=int,
)
@click.help_option("-h", "--help")
def configure_cli(**kwargs) -> None:
    """
    Configure Montreal Forced Aligner command lines to new defaults

    """
    if kwargs.get("profile", None) is not None:
        config.GLOBAL_CONFIG.current_profile_name = kwargs.pop("profile")
    config.GLOBAL_CONFIG.current_profile.update(kwargs)
    config.GLOBAL_CONFIG.save()
