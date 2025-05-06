"""
Preprocess Moroccan dialect datasets for fine-tuning.
"""
import os
import json
import argparse
import glob
import pandas as pd
from tqdm import tqdm

def process_doda_dataset(input_dir, output_file):
    """
    Process the Darija Open Dataset (DODa).
    
    Args:
        input_dir: Input directory containing the DODa dataset
        output_file: Output JSON file path
    """
    print(f"Processing DODa dataset from {input_dir}")
    
    # Find all JSON files in the dataset
    json_files = glob.glob(os.path.join(input_dir, "**/*.json"), recursive=True)
    
    data = []
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                
            if isinstance(file_data, dict):
                # Process dictionary format
                for key, value in file_data.items():
                    if isinstance(value, dict) and 'darija' in value and 'english' in value:
                        data.append({
                            "text": value['darija'],
                            "translation": value['english']
                        })
            elif isinstance(file_data, list):
                # Process list format
                for item in file_data:
                    if isinstance(item, dict) and 'darija' in item and 'english' in item:
                        data.append({
                            "text": item['darija'],
                            "translation": item['english']
                        })
        except Exception as e:
            print(f"Error processing file {json_file}: {e}")
    
    # Save processed data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {len(data)} entries, saved to {output_file}")
    return data

def process_moroccan_darija_datasets(input_dir, output_file):
    """
    Process the Moroccan Darija Datasets collection.
    
    Args:
        input_dir: Input directory containing the datasets
        output_file: Output JSON file path
    """
    print(f"Processing Moroccan Darija datasets from {input_dir}")
    
    data = []
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(input_dir, "**/*.csv"), recursive=True)
    
    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
            
            # Try to identify columns containing Darija text
            text_columns = [col for col in df.columns if any(keyword in col.lower() 
                           for keyword in ['text', 'content', 'darija', 'arabic', 'comment'])]
            
            if text_columns:
                for _, row in df.iterrows():
                    text = row[text_columns[0]]
                    if isinstance(text, str) and text.strip():
                        entry = {"text": text.strip()}
                        
                        # Try to find translation if available
                        translation_cols = [col for col in df.columns if any(keyword in col.lower() 
                                          for keyword in ['translation', 'english', 'en'])]
                        
                        if translation_cols and pd.notna(row[translation_cols[0]]):
                            entry["translation"] = row[translation_cols[0]]
                        
                        data.append(entry)
        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")
    
    # Save processed data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {len(data)} entries, saved to {output_file}")
    return data

def merge_datasets(datasets, output_file):
    """
    Merge multiple datasets into a single file.
    
    Args:
        datasets: List of datasets to merge
        output_file: Output JSON file path
    """
    merged_data = []
    
    for dataset in datasets:
        merged_data.extend(dataset)
    
    # Remove duplicates
    unique_data = []
    seen_texts = set()
    
    for item in merged_data:
        text = item["text"]
        if text not in seen_texts:
            seen_texts.add(text)
            unique_data.append(item)
    
    # Save merged data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, ensure_ascii=False, indent=2)
    
    print(f"Merged dataset contains {len(unique_data)} unique entries")
    return unique_data

def main():
    parser = argparse.ArgumentParser(description="Preprocess Moroccan dialect datasets")
    parser.add_argument("--input_dir", type=str, default="./datasets", help="Directory containing raw datasets")
    parser.add_argument("--output_dir", type=str, default="./processed", help="Directory to save processed datasets")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process DODa dataset
    doda_dir = os.path.join(args.input_dir, "darija_open_dataset")
    doda_output = os.path.join(args.output_dir, "doda_processed.json")
    doda_data = process_doda_dataset(doda_dir, doda_output)
    
    # Process other Moroccan Darija datasets
    other_dir = os.path.join(args.input_dir, "moroccan_darija")
    other_output = os.path.join(args.output_dir, "moroccan_darija_processed.json")
    other_data = process_moroccan_darija_datasets(other_dir, other_output)
    
    # Merge all datasets
    merged_output = os.path.join(args.output_dir, "moroccan_dialects.json")
    merge_datasets([doda_data, other_data], merged_output)

if __name__ == "__main__":
    main()