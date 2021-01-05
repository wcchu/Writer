#!/usr/bin/env bash

# training
echo "training model..."
python learn.py

# deployment
echo "deploying model locally..."
python main.py

echo "done"
