# Montreal Forced Aligner

![Continuous Integration](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/actions/workflows/main.yml/badge.svg)
[![codecov](https://codecov.io/gh/MontrealCorpusTools/Montreal-Forced-Aligner/branch/main/graph/badge.svg?token=GgfM9GXFJ4)](https://codecov.io/gh/MontrealCorpusTools/Montreal-Forced-Aligner)
[![Documentation Status](https://readthedocs.org/projects/montreal-forced-aligner/badge/?version=latest)](http://montreal-forced-aligner.readthedocs.io/en/latest/?badge=latest)
[![Interrogate Status](https://montreal-forced-aligner.readthedocs.io/en/latest/_static/interrogate_badge.svg)](https://github.com/MontrealCorpusTools/montreal-forced-aligner/)
[![DOI](https://zenodo.org/badge/44983969.svg)](https://zenodo.org/badge/latestdoi/44983969)

The Montreal Forced Aligner is a command line utility for performing forced alignment of speech datasets using Kaldi (http://kaldi-asr.org/).

Please see the documentation http://montreal-forced-aligner.readthedocs.io for installation and usage.

If you run into any issues, please check the [mailing list](https://groups.google.com/forum/#!forum/mfa-users) for fixes/workarounds or to post a [new issue](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/issues).

## Installation

You can install MFA either entirely through [conda](https://docs.conda.io/en/latest/) or a mix of conda for Kaldi and Pynini dependencies and Python packaging for MFA itself

### Conda installation

MFA is hosted on [conda-forge](https://conda-forge.org/) and can be installed via:

```
conda install -c conda-forge montreal-forced-aligner
```

in your environment of choice.

### Source installation

If you'd like to install a local version of MFA or want to use the development set up, the easiest way is first create the dev environment from the yaml in the repo root directory:

```
conda env create -n mfa-dev -f environment.yml
```

Alternatively, the dependencies can be installed via:

```
conda install -c conda-forge python=3.8 kaldi sox librosa biopython praatio tqdm requests colorama pyyaml pynini openfst baumwelch ngram
```

MFA can be installed in develop mode via:

```
pip install -e .[dev]
```

You should be able to see appropriate output from `mfa version`

#### Development

The test suite is run via `tox -e py38-win` or `tox -e py38-unix` depending on the OS, and the docs are generated via `tox -e docs`


## Quick links

* [Getting started docs](https://montreal-forced-aligner.readthedocs.io/en/latest/getting_started.html)
* [User Guide](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/index.html)
* [API Reference](https://montreal-forced-aligner.readthedocs.io/en/latest/reference/index.html)
* [Release notes](https://montreal-forced-aligner.readthedocs.io/en/latest/changelog/index.html)
* [MFA Models](https://github.com/MontrealCorpusTools/mfa-models)
* [Eleanor Chodroff's MFA tutorial](https://lingmethodshub.github.io/content/tools/mfa/mfa-tutorial/)
* [@mmcauliffe's forced alignment blog posts](https://memcauliffe.com/tag/forced-alignment.html)
