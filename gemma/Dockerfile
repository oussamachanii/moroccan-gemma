FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# System packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /gemma

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy model code
COPY . .

# Set up model serving
EXPOSE 8080

# Run the model server
CMD ["python3", "server.py"]