FROM python:3.10-slim

WORKDIR /chroma

# Create data directory
RUN mkdir -p /chroma/data

# Copy the server script
COPY scripts/server.py /chroma/server.py

# Expose port
EXPOSE 8000

# Run the mock server
CMD ["python", "/chroma/server.py"]