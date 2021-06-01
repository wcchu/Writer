#!/usr/bin/env bash

sleep 2h

now=$(date +"%F-%T")

# set up gcloud
export PATH=/usr/local/google-cloud-sdk/bin:$PATH
. /.bashrc
gcloud auth activate-service-account --key-file=/secrets/writer-app-engine.json
gcloud config set core/disable_usage_reporting true
gcloud config set component_manager/disable_update_check true
gcloud config set project writer-01

# train model
echo "training model..."
python learn.py

# upload model
echo "uploading model..."
gsutil cp *.pk gs://writer-training/${now}/
gsutil cp -r checkpoints gs://writer-training/${now}/

# deploy to google app engine
gcloud app deploy

# if deploying locally:
# python main.py

echo "done"
