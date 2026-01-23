# Use a python base image
FROM python:3.11-slim

# Install system dependencies (build-essential for spacy/transformers if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the application and models
COPY . /app
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Pre-download models to bake them into the image
RUN python download_models.py && rm download_models.py

# Set the command to run the production script
CMD ["python", "analysis.py"]
