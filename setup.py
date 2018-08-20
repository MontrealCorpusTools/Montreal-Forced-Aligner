import sys
import os
from setuptools import setup
from setuptools.command.test import test as TestCommand


def readme():
    with open('README.md') as f:
        return f.read()


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['--strict', '--verbose', '--tb=long', '--skiplarge', 'tests']
        #if os.environ.get('TRAVIS', False):
        #    self.test_args = ['-x', '--strict', '--verbose', '--tb=long', 'tests']
        self.test_suite = True

    def run_tests(self):
        if __name__ == '__main__':  # Fix for multiprocessing infinite recursion on Windows
            import pytest
            errcode = pytest.main(self.test_args)
            sys.exit(errcode)


if __name__ == '__main__':
    setup(name='Montreal Forced Aligner',
          version='1.0.0',
          description='',
          long_description='',
          classifiers=[
              'Development Status :: 3 - Alpha',
              'Programming Language :: Python',
              'Programming Language :: Python :: 3',
              'Operating System :: OS Independent',
              'Topic :: Scientific/Engineering',
              'Topic :: Text Processing :: Linguistic',
          ],
          keywords='phonology corpus phonetics alignment segmentation',
          url='https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner',
          author='Montreal Corpus Tools',
          author_email='michael.e.mcauliffe@gmail.com',
          packages=['aligner',
                    'aligner.aligner',
                    'aligner.g2p',
                    'aligner.command_line',
                    'aligner.config',
                    'aligner.features',
                    'aligner.trainers',
                    'aligner.gui'],
          install_requires=[
              'textgrid',
              'tqdm',
              'alignment',
              'pyyaml',
          ],
          package_data={'aligner.config': ['*.yaml']},
          cmdclass={'test': PyTest},
          extras_require={
              'testing': ['pytest'],
          }
          )
