#!/usr/bin/env bash

# training
echo "training model..."
python learn.py

# deployment
echo "deploying model locally (port = 5000)"
python main.py

echo "done"
