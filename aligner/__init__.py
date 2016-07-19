__ver_major__ = 0
__ver_minor__ = 5
__ver_patch__ = 0
__ver_tuple__ = (__ver_major__, __ver_minor__, __ver_patch__)
__version__ = "%d.%d.%d" % __ver_tuple__

__all__ = ['aligner', 'command_line', 'archive', 'corpus', 'config', 'dictionary', 'exceptions',
            'helper', 'multiprocessing', 'textgrid', 'utils']

import .aligner as aligner

import .command_line as command_line

import .archive as archive

import .corpus as corpus

import .dictionary as dictionary

import .exceptions as exceptions

import .helper as helper

import .config as config

import .multiprocessing as multiprocessing

import .textgrid as textgrid

import .utils as utils
