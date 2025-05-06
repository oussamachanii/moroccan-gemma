# Moroccan Dialect Gemma 3 AI with ChromaDB

This project creates a containerized environment to fine-tune and run Google's Gemma 3 language model on Moroccan dialectal Arabic (Darija) using ChromaDB for efficient vector storage and retrieval.

## Features

- Fine-tunes Gemma 3 on Moroccan Darija dialect
- Uses ChromaDB for semantic search of Darija text
- Fully containerized with Docker
- Provides an interactive web UI for chatting with the model
- Includes scripts for downloading and processing Moroccan dialect datasets
- Supports both Darija and English responses

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended for faster training)
- Docker NVIDIA Container Toolkit (for GPU support)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/moroccan-gemma.git
cd moroccan-gemma
```

2. Run the setup script:
```bash
chmod +x run.sh
./run.sh
```

3. Access the web interface at http://localhost:5000

## Project Structure

- `app/`: Flask web application and API
- `gemma/`: Gemma 3 model server and fine-tuning code
- `chroma/`: ChromaDB vector database
- `data/`: Moroccan dialect datasets and preprocessing scripts

## Customization

### Using Your Own Dataset

Place your Moroccan dialect data in the `data/` directory and modify the preprocessing scripts as needed.

### Model Configuration

You can configure the model parameters in the `.env` file:

```
# Change to a different Gemma model size
GEMMA_MODEL_NAME=google/gemma-3-4b
```

## Fine-tuning

1. Prepare your dataset using the preprocessing scripts in `data/`.
2. Access the web UI and click the "Fine-tune Model" button.
3. Alternatively, trigger fine-tuning via the API:
```bash
curl -X POST http://localhost:5000/api/fine-tune
```

## Acknowledgements

- [Google Gemma AI](https://ai.google.dev/gemma) for the open-source Gemma 3 model
- [ChromaDB](https://www.trychroma.com/) for the vector database
- The contributors to various Moroccan Darija datasets, including DODa (Darija Open Dataset)

## License

This project is licensed under the MIT License - see the LICENSE file for details.