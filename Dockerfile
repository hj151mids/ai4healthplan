# Use the official lightweight Python 3.10 image as a base
FROM python:3.10-slim

# Prevent Python from writing .pyc files and ensure logs are sent straight to terminal
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system-level dependencies required for high-performance ML libraries
# - build-essential: Necessary for compiling certain Python extensions
# - libgomp1: Critical dependency for XGBoost's multi-threaded execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt first to leverage Docker's layer caching
# This speeds up subsequent builds if dependencies haven't changed
COPY requirements.txt .

# Upgrade pip and install all project dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Google Cloud Run expects the container to listen on port 8080 by default
EXPOSE 8080

# Start the FastAPI application using uvicorn
# --host 0.0.0.0: Required for the container to be reachable from outside
# --port 8080: Aligns with Cloud Run's default ingress port
# --workers 1: Standard for single-concurrency or low-memory serverless instances
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
