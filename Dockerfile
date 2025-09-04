# Use Python 3.11 slim image as base (matches pyproject.toml requirements)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy pyproject.toml and setup files
COPY pyproject.toml .
COPY README.md .

# Copy application code
COPY src/ src/
COPY utils/ utils/
COPY config/ config/

# Copy main scripts
COPY fantasy_football_multi_league.py .
COPY lineup_optimizer.py .
COPY matchup_analyzer.py .

# Create necessary directories for runtime
RUN mkdir -p /app/logs /app/cache

# Create a non-root user to run the application
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set Python path
ENV PYTHONPATH=/app

# Expose port if the MCP server needs it (adjust as needed)
EXPOSE 8000

# Default command to run the MCP server
CMD ["python", "-m", "src.mcp_server"]