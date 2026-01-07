# Multi-stage build for optimized image size
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY setup.py .
COPY README.md .

# Make sure scripts are in PATH
ENV PATH=/root/.local/bin:$PATH

# Create necessary directories
RUN mkdir -p data/raw data/processed models logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port for API (if needed)
EXPOSE 8000

# Default command
CMD ["python", "scripts/run_pipeline.py"]
