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
  conda create -q -n test-environment python=$TRAVIS_PYTHON_VERSION pytest setuptools
  source activate test-environment
  which python
  pip install -q coveralls coverage textgrid tqdm
else
  echo "Miniconda already installed."
fi

cd $HOME/build/MontrealCorpusTools/Montreal-Forced-Aligner
source activate test-environment
pip install -r requirements.txt
python thirdparty/download_binaries.py $HOME/tools --keep --redownload

if [ ! -d "$HOME/tools/mfa_test_data" ]; then
  cd $HOME/tools
  git clone https://github.com/MontrealCorpusTools/mfa_test_data.git
else
  cd $HOME/tools/mfa_test_data
  git pull origin
  echo "Test data already installed."
fi

