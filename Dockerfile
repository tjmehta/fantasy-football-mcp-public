# Use Python 3.11 slim image as base (matches pyproject.toml requirements)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user first
RUN useradd -m -u 1000 appuser

# Copy requirements and install dependencies as root for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY --chown=appuser:appuser pyproject.toml README.md ./
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser utils/ ./utils/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser fantasy_football_multi_league.py lineup_optimizer.py matchup_analyzer.py ./

# Create necessary directories with correct ownership
RUN mkdir -p /app/logs /app/cache && \
    chown -R appuser:appuser /app/logs /app/cache

# Verify files were copied (for debugging)
RUN echo "=== Listing /app contents ===" && \
    ls -la /app/ && \
    echo "=== Listing /app/src contents ===" && \
    ls -la /app/src/ || echo "src directory not found"

# Switch to non-root user
USER appuser

# Expose port if the MCP server needs it
EXPOSE 8000

# Default command to run the MCP server
CMD ["python", "-m", "src.mcp_server"]