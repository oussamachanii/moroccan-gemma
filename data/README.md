# Moroccan Dialect Datasets

This directory contains scripts and data for Moroccan dialect (Darija) processing.

## Dataset Sources

The project uses the following datasets for fine-tuning:

1. **Darija Open Dataset (DODa)** - A collection of Moroccan dialect text with English translations
   - Source: https://github.com/darija-open-dataset/darija-open-dataset

2. **Moroccan Darija Dataset Collection** - Various datasets of Moroccan dialect text
   - Source: https://github.com/nainiayoub/moroccan-darija-datasets

## Directory Structure

```
data/
   download_datasets.py       # Script to download raw datasets
   preprocess.py              # Script to process and clean datasets
   populate_chroma.py         # Script to populate ChromaDB with processed data
   datasets/                  # Raw downloaded datasets
   processed/                 # Processed and cleaned datasets
```

## Usage

### 1. Download Datasets

To download the datasets, run:

```bash
python download_datasets.py --output_dir ./datasets
```

### 2. Preprocess Datasets

To preprocess the downloaded datasets, run:

```bash
python preprocess.py --input_dir ./datasets --output_dir ./processed
```

### 3. Populate ChromaDB

To populate ChromaDB with the processed data, run:

```bash
python populate_chroma.py --input_file ./processed/moroccan_dialects.json
```

## Dataset Format

The processed dataset is stored in JSON format with the following structure:

```json
[
  {
    "text": "Moroccan dialect text",
    "translation": "English translation (if available)"
  },
  ...
]
```

## Extending the Dataset

To add more data to the dataset:

1. Add new data sources to the `DATASET_URLS` dictionary in `download_datasets.py`
2. Implement appropriate processing in `preprocess.py`
3. Run the scripts to download and process the new data

## Note

The quality of the fine-tuned model depends on the quality and quantity of the training data. More diverse and high-quality Moroccan dialect data will result in better model performance.