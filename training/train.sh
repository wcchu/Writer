#!/usr/bin/env bash

TIME=$(date +"%F-%T")
GSDIR="gs://writer-training"

# set up gcloud
export PATH=/usr/local/google-cloud-sdk/bin:$PATH
. .bashrc
gcloud config set core/disable_usage_reporting true
gcloud config set component_manager/disable_update_check true
gcloud config set project writer-01
gcloud auth activate-service-account --key-file=writer-app-engine.json

# training
echo "training model..."
python learn.py

# upload model
echo "uploading model..."
gsutil -m cp *.pk ${GSDIR}/${TIME}/
gsutil -m cp -r ${GSDIR}/${TIME}/* ${GSDIR}/latest/

echo "done"
