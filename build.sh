#!/bin/bash
# Render build script - runs during deployment
set -o errexit

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Build completed successfully!"
