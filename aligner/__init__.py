__ver_major__ = 0
__ver_minor__ = 5
__ver_patch__ = 0
__ver_tuple__ = (__ver_major__, __ver_minor__, __ver_patch__)
__version__ = "%d.%d.%d" % __ver_tuple__

__all__ = ['aligner', 'command_line', 'archive', 'corpus', 'config', 'dictionary', 'exceptions',
            'helper', 'multiprocessing', 'textgrid', 'utils']

import aligner.aligner as aligner

import aligner.command_line as command_line

import aligner.archive as archive

import aligner.corpus as corpus

import aligner.dictionary as dictionary

import aligner.exceptions as exceptions

import aligner.helper as helper

import aligner.config as config

import aligner.multiprocessing as multiprocessing

import aligner.textgrid as textgrid

import aligner.utils as utils
