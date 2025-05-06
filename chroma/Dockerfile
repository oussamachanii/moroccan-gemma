FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /chroma

# Install Chroma
RUN pip install --no-cache-dir chromadb==0.4.18

# Create data directory for persistence
RUN mkdir -p /chroma/data

# Copy configuration
COPY config /chroma/config

# Expose Chroma API port
EXPOSE 8000

# Start Chroma server
CMD ["python", "-m", "chromadb.server", "--host", "0.0.0.0", "--port", "8000"]