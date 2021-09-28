import os

from ..exceptions import ArgumentError
from ..thirdparty.download import download_binaries
from ..thirdparty.kaldi import collect_kaldi_binaries, validate_kaldi_binaries


def validate_args(args):
    available_commands = ['download', 'validate', 'kaldi']
    if args.command not in available_commands:
        raise ArgumentError(
            '{} is not a valid thirdparty command ({})'.format(args.command, ', '.join(available_commands)))
    if args.command not in ['download', 'validate']:
        if not args.local_directory:
            raise ArgumentError('Specify a directory to extract {} binaries from.'.format(args.command))
        if not os.path.exists(args.local_directory):
            raise ArgumentError('The directory {} does not exist.'.format(args.local_directory))


def run_thirdparty(args):
    validate_args(args)
    if args.command == 'download':
        download_binaries()
    elif args.command == 'validate':
        validate_kaldi_binaries()
    elif args.command == 'kaldi':
        collect_kaldi_binaries(args.local_directory)
