#!/bin/bash

# Build script for DigitalOcean deployment

echo "Building frontend assets..."
npm install
npm run build

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Building complete!"
