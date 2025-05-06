#!/bin/bash

# Script to apply ChromaDB configuration

# Create necessary directories
mkdir -p /chroma/data/logs

# Copy configuration files to the correct locations
if [ -f /chroma/config/chroma_server_config.yaml ]; then
    echo "Applying ChromaDB server configuration..."
    cp /chroma/config/chroma_server_config.yaml /etc/chromadb/chroma_server_config.yaml
fi

if [ -f /chroma/config/logconfig.yaml ]; then
    echo "Applying ChromaDB logging configuration..."
    cp /chroma/config/logconfig.yaml /etc/chromadb/logconfig.yaml
fi

# Run initialization script
if [ -f /chroma/config/chroma_init.py ]; then
    echo "Running ChromaDB initialization script..."
    python /chroma/config/chroma_init.py
fi

echo "ChromaDB configuration update complete."