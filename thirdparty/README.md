

Thirdparty requirements
=======================


Windows
-------

Versions:

```
kaldi===094d22746b604fd20c2b8730966c9d0bc9f2170b (08/18/2017)
OpenFst==1.6.9
OpenGrm-Ngram==1.3.4
Phonetisaurus==64719ca40c17cb70d810fffadac52c97984ca539 (07/16/2017)
```

Linux
-----

Versions:

```
kaldi===094d22746b604fd20c2b8730966c9d0bc9f2170b (08/18/2017)
OpenFst==1.6.9
OpenGrm-Ngram==1.3.4
Phonetisaurus==64719ca40c17cb70d810fffadac52c97984ca539 (07/16/2017)
```

Instructions:

Paths are based on the build system that @mmcauliffe has (Ubuntu 16.04 on Windows 10).

Kaldi and openfst
(bumped openfst version to 1.6.9 from 1.6.7)
```
git clone https://github.com/kaldi-asr/kaldi.git
cd kaldi/tools
extras/check_dependencies.sh
make -j 6
make openblas
cd ../src
./configure --shared --openblas-root=../tools/OpenBLAS/install
make depend -j 6
make -j 6
```

OpenGRM

```
export LD_LIBRARY_PATH=/mnt/e/Dev/Linux/kaldi/tools/openfst/lib
export CPLUS_INCLUDE_PATH=/mnt/e/Dev/Linux/kaldi/tools/openfst/src/include
./configure --prefix=`pwd`/install
make
make install
```

Phonetisaurus

```
git clone https://github.com/AdolfVonKleist/Phonetisaurus.git
cd Phonetisaurus
./configure --enable-static=no --with-openfst-includes=/mnt/e/Dev/Linux/kaldi/tools/openfst/include --with-openfst-libs=/mnt/e/Dev/Linux/kaldi/tools/openfst/lib
make
```

Collecting binaries

From MFA root


Mac
---

```
kaldi===094d22746b604fd20c2b8730966c9d0bc9f2170b (08/18/2017)
OpenFst==1.6.9
OpenGrm-Ngram==1.3.4
Phonetisaurus==64719ca40c17cb70d810fffadac52c97984ca539 (07/16/2017)

```

Instructions:

Paths are based on the build system that @mmcauliffe has.

Kaldi and openfst
(bumped openfst version to 1.6.9 from 1.6.7)

```
brew install automake autoconf python@2 libtool gcc
git clone https://github.com/kaldi-asr/kaldi.git
cd kaldi/tools
extras/check_dependencies.sh
make -j 6
make openblas
cd ../src
./configure --shared --openblas-root=../tools/OpenBLAS/install
make depend -j 6
make -j 6
```

OpenGRM

```
export LD_LIBRARY_PATH=/mnt/e/Dev/Linux/kaldi/tools/openfst/lib
export CPLUS_INCLUDE_PATH=/mnt/e/Dev/Linux/kaldi/tools/openfst/src/include
wget http://www.opengrm.org/twiki/pub/GRM/NGramDownload/opengrm-ngram-1.3.4.tar.gz
tar -xvzf opengrm-ngram-1.3.4.tar.gz
./configure --prefix=`pwd`/install
make
make install
```

Phonetisaurus

```
git clone https://github.com/AdolfVonKleist/Phonetisaurus.git
cd Phonetisaurus
./configure --enable-static=no --with-openfst-includes=/mnt/e/Dev/Linux/kaldi/tools/openfst/include --with-openfst-libs=/mnt/e/Dev/Linux/kaldi/tools/openfst/lib
make
```