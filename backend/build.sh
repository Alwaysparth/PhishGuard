#!/usr/bin/env bash
# Render build script — runs before the server starts
set -e

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Creating required directories..."
mkdir -p models data

echo "Build complete."
