import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API configurations
GEMMA_API_URL = os.getenv("GEMMA_API_URL", "http://gemma:8080")
CHROMA_API_URL = os.getenv("CHROMA_API_URL", "http://chroma:8000")

# Flask app configurations
DEBUG_MODE = os.getenv("FLASK_ENV", "development") == "development"
SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key-for-dev")

# Model configurations
GEMMA_MODEL_NAME = os.getenv("GEMMA_MODEL_NAME", "google/gemma-3-1b")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/root/.cache/huggingface")

# ChromaDB configurations
CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = "moroccan_dialect"