#!/usr/bin/env bash

now=$(date +%s)

# train model
echo "training model..."
python learn.py

# upload model
echo "uploading model..."
gsutil cp *.pk gs://writer-training/${now}/
gsutil cp -r checkpoints gs://writer-training/${now}/

echo "done"
