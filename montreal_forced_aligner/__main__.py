from rich.traceback import install

from montreal_forced_aligner.command_line.mfa import mfa_cli

install(show_locals=True)
mfa_cli()
