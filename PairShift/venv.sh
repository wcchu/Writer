#!/usr/bin/env bash
prefix=python-env
rm -r $prefix
virtualenv $prefix
source $prefix/bin/activate
easy_install -U pip
pip install --upgrade setuptools
pip install --upgrade cython
pip install oauth2client httplib2 numpy protobuf pytest sklearn scipy pandas matplotlib cython pyreadr flake8 yapf
pip install tensorflow==2.0.0-beta1
# integrate with jupyter
pip install ipykernel
pip install library
pip install jupyter
