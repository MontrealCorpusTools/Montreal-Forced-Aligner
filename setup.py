import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand


def readme():
    with open('README.md') as f:
        return f.read()


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['--strict', '--verbose', '--tb=long', 'tests']
        self.test_suite = True

    def run_tests(self):
        if __name__ == '__main__':  # Fix for multiprocessing infinite recursion on Windows
            import pytest
            errcode = pytest.main(self.test_args)
            sys.exit(errcode)


if __name__ == '__main__':
    setup(name='Montreal Forced Aligner',
          version='2.0.0a0',
          description='Montreal Forced Aligner is a package for aligning speech corpora through the use of '
                      'acoustic models and dictionaries using Kaldi functionality.',
          long_description=readme(),
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
          packages=['montreal_forced_aligner',
                    'montreal_forced_aligner.aligner',
                    'montreal_forced_aligner.command_line',
                    'montreal_forced_aligner.config',
                    'montreal_forced_aligner.corpus',
                    'montreal_forced_aligner.features',
                    'montreal_forced_aligner.g2p',
                    'montreal_forced_aligner.gui',
                    'montreal_forced_aligner.lm',
                    'montreal_forced_aligner.thirdparty',
                    'montreal_forced_aligner.trainers'],
          install_requires=[
              'textgrid',
              'numpy',
              'tqdm',
              'pyyaml',
              'librosa',
              'pyqt5',
              'pyqtgraph',
              'requests',
          ],
          python_requires='>=3.8',
          entry_points={
              'console_scripts': ['mfa=montreal_forced_aligner.command_line.mfa:main']
          },
          package_data={'montreal_forced_aligner.config': ['*.yaml']},
          cmdclass={'test': PyTest},
          extras_require={
              'testing': ['pytest'],
          }
          )
