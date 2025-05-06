#!/bin/bash

# Welcome message
echo "Starting Moroccan Dialect Gemma 3 AI with ChromaDB..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed or not in PATH."
    exit 1
fi

# Create data directories if they don't exist
mkdir -p data/datasets
mkdir -p data/processed
mkdir -p chroma/data

# Download and preprocess datasets
echo "Do you want to download and preprocess the Moroccan dialect datasets? (y/n)"
read -r download_datasets

if [ "$download_datasets" = "y" ] || [ "$download_datasets" = "Y" ]; then
    echo "Downloading datasets..."
    docker-compose run --rm app python /data/download_datasets.py
    
    echo "Preprocessing datasets..."
    docker-compose run --rm app python /data/preprocess.py
    
    echo "Populating ChromaDB..."
    docker-compose run --rm app python /data/populate_chroma.py
fi

# Start the containers
echo "Starting Docker containers..."
docker-compose up -d

echo "
---------------------------------------------------
Moroccan Dialect Gemma 3 AI with ChromaDB is running!
---------------------------------------------------

Web interface:  http://localhost:5000
ChromaDB API:   http://localhost:8000
Gemma API:      http://localhost:8080

To stop the application, run:
    docker-compose down

For logs, run:
    docker-compose logs -f
"