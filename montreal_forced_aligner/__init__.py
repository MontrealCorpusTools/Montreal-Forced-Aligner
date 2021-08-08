__ver_major__ = 2
__ver_minor__ = 0
__ver_patch__ = '0a23'
__version__ = "{}.{}.{}".format(__ver_major__, __ver_minor__, __ver_patch__)

__all__ = ['aligner', 'command_line', 'models', 'corpus', 'config', 'dictionary', 'exceptions',
            'helper', 'multiprocessing', 'textgrid', 'g2p', '__version__']

import montreal_forced_aligner.aligner as aligner

import montreal_forced_aligner.command_line as command_line

import montreal_forced_aligner.models as models

import montreal_forced_aligner.corpus as corpus

import montreal_forced_aligner.dictionary as dictionary

import montreal_forced_aligner.exceptions as exceptions

import montreal_forced_aligner.helper as helper

import montreal_forced_aligner.config as config

import montreal_forced_aligner.multiprocessing as multiprocessing

import montreal_forced_aligner.textgrid as textgrid

import montreal_forced_aligner.g2p as g2p
