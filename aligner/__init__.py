__ver_major__ = 1
__ver_minor__ = 0
__ver_patch__ = 0
__ver_tuple__ = (__ver_major__, __ver_minor__, __ver_patch__)
__version__ = "%d.%d.%d" % __ver_tuple__

__all__ = ['aligner', 'command_line', 'models', 'corpus', 'config', 'dictionary', 'exceptions',
            'helper', 'multiprocessing', 'textgrid', 'g2p']

import aligner.aligner as aligner

import aligner.command_line as command_line

import aligner.models as models

import aligner.corpus as corpus

import aligner.dictionary as dictionary

import aligner.exceptions as exceptions

import aligner.helper as helper

import aligner.config as config

import aligner.multiprocessing as multiprocessing

import aligner.textgrid as textgrid

import aligner.g2p as g2p
