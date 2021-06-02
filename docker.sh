#!/usr/bin/env bash

TIME=$(date +"%F-%T")

# set up gcloud
export PATH=/usr/local/google-cloud-sdk/bin:$PATH
. /.bashrc
gcloud auth activate-service-account --key-file=/secrets/writer-service.json
gcloud config set core/disable_usage_reporting true
gcloud config set component_manager/disable_update_check true

# train model
echo "training model..."
python learn.py

# upload model
echo "uploading model..."
gsutil cp *.pk gs://writer-training/${TIME}/
gsutil cp -r checkpoints gs://writer-training/${TIME}/

# deploy to google app engine
gcloud app deploy

# if deploying locally:
# python main.py

echo "done"
