#!/bin/sh
set -e

#check to see if miniconda folder is empty
if [ ! -d "$HOME/miniconda/miniconda/envs/test-environment" ]; then
  wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  chmod +x miniconda.sh
  ./miniconda.sh -b -p $HOME/miniconda/miniconda
  export PATH="$HOME/miniconda/miniconda/bin:$PATH"
  conda config --set always_yes yes --set changeps1 no
  conda update -q conda
  conda info -a
  conda create -q -n test-environment python=$TRAVIS_PYTHON_VERSION scipy pytest setuptools
  source activate test-environment
  which python
  pip install -q coveralls coverage textgrid tqdm
else
  echo "Miniconda already installed."
fi

if [ ! -d "$HOME/tools/kaldi" ]; then
  mkdir -p $HOME/downloads
  cd $HOME/downloads
  git clone https://github.com/kaldi-asr/kaldi.git kaldi --origin upstream
  cd kaldi/tools
  extras/check_dependencies.sh
  make -j 4
  cd ../src
  ./configure
  make depend -j 4
  make -j 4
  cd $HOME/build/MontrealCorpusTools/Montreal-Forced-Aligner
  source activate test-environment
  python thirdparty/kaldibinaries.py $HOME/downloads/kaldi
  cp -R thirdparty/bin $HOME/tools/kaldi
else
  echo "Kaldi already installed."
fi

if [ ! -d "$HOME/tools/mfa_test_data" ]; then
  mkdir -p $HOME/downloads
  cd $HOME/tools
  git clone https://github.com/MontrealCorpusTools/mfa_test_data.git
else
  echo "Test data already installed."
fi
