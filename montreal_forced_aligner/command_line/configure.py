import os

import click

from montreal_forced_aligner.config import GLOBAL_CONFIG, MFA_PROFILE_VARIABLE

__all__ = ["configure_cli"]


@click.command(
    "configure",
    help="The configure command is used to set global defaults for MFA so "
    "you don't have to set them every time you call an MFA command.",
)
@click.option(
    "-p",
    "--profile",
    help='Configuration profile to use, defaults to "global"',
    type=str,
    default=None,
)
@click.option(
    "--temporary_directory",
    "-t",
    help=f"Set the default temporary directory, default is {GLOBAL_CONFIG.temporary_directory}",
    type=str,
    default=None,
)
@click.option(
    "--num_jobs",
    "-j",
    help=f"Set the number of processes to use by default, defaults to {GLOBAL_CONFIG.num_jobs}",
    type=int,
    default=None,
)
@click.option(
    "--always_clean/--never_clean",
    "clean",
    help="Turn on/off clean mode where MFA will clean temporary files before each run.",
    default=None,
)
@click.option(
    "--always_verbose/--never_verbose",
    "verbose",
    help="Turn on/off verbose mode where MFA will print more output.",
    default=None,
)
@click.option(
    "--always_quiet/--never_quiet",
    "quiet",
    help="Turn on/off quiet mode where MFA will not print any output.",
    default=None,
)
@click.option(
    "--always_debug/--never_debug",
    "debug",
    help="Turn on/off extra debugging functionality.",
    default=None,
)
@click.option(
    "--always_overwrite/--never_overwrite",
    "overwrite",
    help="Turn on/off overwriting export files.",
    default=None,
)
@click.option(
    "--enable_mp/--disable_mp",
    "use_mp",
    help="Turn on/off multiprocessing. Multiprocessing is recommended will allow for faster executions.",
    default=None,
)
@click.option(
    "--enable_textgrid_cleanup/--disable_textgrid_cleanup",
    "cleanup_textgrids",
    help="Turn on/off post-processing of TextGrids that cleans up "
    "silences and recombines compound words and clitics.",
    default=None,
)
@click.option(
    "--enable_terminal_colors/--disable_terminal_colors",
    "terminal_colors",
    help="Turn on/off colored text in command line output.",
    default=None,
)
@click.option(
    "--blas_num_threads",
    help="Number of threads to use for BLAS libraries, 1 is recommended "
    "due to how much MFA relies on multiprocessing. "
    f"Currently set to {GLOBAL_CONFIG.blas_num_threads}.",
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
    "--database_port",
    default=None,
    help="Port for postgresql database.",
    type=int,
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
        os.putenv(MFA_PROFILE_VARIABLE, kwargs["profile"])
    GLOBAL_CONFIG.current_profile.update(kwargs)
    GLOBAL_CONFIG.save()
