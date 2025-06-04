FROM python:3.11-slim

# Set Hugging Face home
ENV HF_HOME=/data/.huggingface
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create primary data directory with proper permissions
RUN mkdir -p /data/processed_data && chmod -R 777 /data

# Copy application
COPY app/ ./app/

# Copy data files to just ONE location - the app's path finding logic will handle it
COPY app/data/processed_data/chunks.pkl data/processed_data/
COPY app/data/processed_data/embedded_docs.pkl data/processed_data/
COPY app/data/processed_data/bm25_retriever.pkl data/processed_data/
COPY app/data/processed_data/embedding_info.json data/processed_data/



# Print debug info to verify file locations
RUN find / -name "chunks.pkl" -o -name "embedded_docs.pkl" 2>/dev/null || echo "Files not found"

COPY check_dependencies.py ./

# Verify setup
RUN python check_dependencies.py

# Enable more verbose logging
ENV PYTHONUNBUFFERED=1

EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--logger.level=debug"]
