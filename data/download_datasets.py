"""
Script to download and organize Moroccan dialect datasets.
"""
import os
import requests
import json
import zipfile
import io
import argparse
from tqdm import tqdm

# URLs for Moroccan Darija datasets
DATASET_URLS = {
    "darija_open_dataset": "https://github.com/darija-open-dataset/darija-open-dataset/archive/refs/heads/master.zip",
    "moroccan_darija": "https://github.com/nainiayoub/moroccan-darija-datasets/archive/refs/heads/main.zip",
}

def download_dataset(url, output_dir):
    """
    Download a dataset from URL and extract it.
    
    Args:
        url: URL to the dataset (ZIP file)
        output_dir: Directory to extract the dataset to
    """
    print(f"Downloading dataset from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Extract zip file
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
        # Extract all contents
        zip_file.extractall(output_dir)
    
    print(f"Dataset downloaded and extracted to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download Moroccan dialect datasets")
    parser.add_argument("--output_dir", type=str, default="./datasets", help="Directory to save datasets")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download each dataset
    for name, url in DATASET_URLS.items():
        dataset_dir = os.path.join(args.output_dir, name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        try:
            download_dataset(url, dataset_dir)
        except Exception as e:
            print(f"Error downloading {name} dataset: {e}")
    
    print("All datasets downloaded successfully!")

if __name__ == "__main__":
    main()