#!/usr/bin/env bash
prefix=python-env
rm -r $prefix
virtualenv $prefix
source $prefix/bin/activate
easy_install -U pip
pip install oauth2client httplib2 numpy protobuf pytest sklearn scipy pandas matplotlib cython pyreadr flake8 yapf pandas-gbq
pip install tensorflow==1.14.0
# integrate with jupyter
pip install ipykernel
pip install library
pip install jupyter
