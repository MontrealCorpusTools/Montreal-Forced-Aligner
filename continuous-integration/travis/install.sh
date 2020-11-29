#!/bin/sh
set -e

if [ ! -d "$HOME/tools/mfa_test_data" ]; then
  cd $HOME/tools
  git clone https://github.com/MontrealCorpusTools/mfa_test_data.git
else
  cd $HOME/tools/mfa_test_data
  git pull origin
  echo "Test data already installed."
fi

