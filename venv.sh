#!/usr/bin/env bash
prefix=env
rm -r $prefix
virtualenv --python=/usr/bin/python3 --system-site-packages $prefix
source $prefix/bin/activate
pip install --ignore-installed --no-cache-dir --upgrade \
        pip \
        numpy \
        pandas \
        sklearn \
        scipy \
        matplotlib \
        flake8 \
        yapf \
        cython \
        testresources \
        jupyter \
        tensorflow==2.0.0-rc0
