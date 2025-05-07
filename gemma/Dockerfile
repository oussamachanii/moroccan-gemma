FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# System packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /gemma

# Set specific versions for PyTorch and Transformers to ensure compatibility
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
# Explicitly install an older version of transformers that's compatible
RUN pip3 install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install transformers==4.30.2 datasets==2.14.5

# Copy model code
COPY . .

# Setup model serving
EXPOSE 8080

# Run the model server
CMD ["python3", "server.py"]