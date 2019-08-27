#!/usr/bin/env bash

# inputs = python filename, pdf filename
enscript -E -q -Z -p - -f Courier10 $1 | ps2pdf - $2
