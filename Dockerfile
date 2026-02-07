# Base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/all-MiniLM-L6-v2

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Embedding Model
COPY download_model.py .
RUN python download_model.py

# Copy application code (to be added later)
# COPY . .

# Expose port
EXPOSE 8000

# Command to run API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
