#!/usr/bin/env bash

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

# deploy to google app engine
echo "deploying service to app engine..."
gcloud app deploy

# deploy to local
echo "deploying service locally..."
python main.py

echo "done"
