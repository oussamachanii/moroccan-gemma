"""
Populate ChromaDB with Moroccan dialect data for semantic search.
"""
import os
import json
import argparse
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def load_dataset(file_path):
    """Load the processed dataset."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def populate_chroma(data, collection_name="moroccan_dialect", host="chroma", port=8000):
    """
    Populate ChromaDB with the dataset.
    
    Args:
        data: List of data items
        collection_name: Name of the ChromaDB collection
        host: ChromaDB host
        port: ChromaDB port
    """
    print(f"Connecting to ChromaDB at {host}:{port}")
    
    # Initialize ChromaDB client
    client = chromadb.HttpClient(
        host=host,
        port=port
    )
    
    # Create or get collection
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Using existing collection: {collection_name}")
    except Exception:
        print(f"Creating new collection: {collection_name}")
        collection = client.create_collection(name=collection_name)
    
    # Load sentence transformer model for embeddings
    print("Loading sentence transformer model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Populate collection
    batch_size = 100
    for i in tqdm(range(0, len(data), batch_size), desc="Adding documents to ChromaDB"):
        batch = data[i:i+batch_size]
        
        ids = [f"doc_{i+j}" for j in range(len(batch))]
        texts = [item["text"] for item in batch]
        metadatas = [{"translation": item.get("translation", "")} for item in batch]
        
        # Generate embeddings
        embeddings = model.encode(texts).tolist()
        
        # Add to collection
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    print(f"Added {len(data)} documents to ChromaDB collection {collection_name}")

def main():
    parser = argparse.ArgumentParser(description="Populate ChromaDB with Moroccan dialect data")
    parser.add_argument("--input_file", type=str, default="./processed/moroccan_dialects.json", help="Processed dataset file")
    parser.add_argument("--host", type=str, default="chroma", help="ChromaDB host")
    parser.add_argument("--port", type=int, default=8000, help="ChromaDB port")
    args = parser.parse_args()
    
    # Load data
    data = load_dataset(args.input_file)
    
    # Populate ChromaDB
    populate_chroma(data, host=args.host, port=args.port)

if __name__ == "__main__":
    main()