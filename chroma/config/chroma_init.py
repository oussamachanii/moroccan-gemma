import os
import yaml
import chromadb
from chromadb.config import Settings

def load_config(file_path):
    """Load YAML configuration file."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def initialize_chroma():
    """Initialize ChromaDB with configurations."""
    # Load server configuration
    config_file = '/chroma/config/chroma_server_config.yaml'
    collections_file = '/chroma/config/collections_config.yaml'
    
    config = load_config(config_file)
    collections_config = load_config(collections_file)
    
    # Initialize client
    persist_directory = config.get('database', {}).get('persist_directory', '/chroma/data')
    
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_directory
    ))
    
    # Create collections if they don't exist
    for collection_name, collection_config in collections_config.get('collections', {}).items():
        try:
            client.get_collection(name=collection_name)
            print(f"Collection '{collection_name}' already exists.")
        except Exception:
            print(f"Creating collection '{collection_name}'...")
            client.create_collection(
                name=collection_name,
                metadata={
                    "description": collection_config.get('description', ''),
                    "schema": collection_config.get('metadata_schema', {})
                }
            )
    
    print("ChromaDB initialization completed.")

if __name__ == "__main__":
    initialize_chroma()