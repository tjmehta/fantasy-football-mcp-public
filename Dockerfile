# Use Python 3.11 slim image as base (matches pyproject.toml requirements)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables (mirroring .env.example defaults)
ENV RUNTIME_ENVIRONMENT=docker \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    CACHE_DIR=./.cache \
    CACHE_TTL_SECONDS=3600 \
    YAHOO_API_RATE_LIMIT=100 \
    YAHOO_API_RATE_WINDOW_SECONDS=3600 \
    LOG_LEVEL=INFO \
    LOG_FILE=./logs/fantasy_football.log \
    MCP_SERVER_NAME=fantasy-football \
    MCP_SERVER_VERSION=1.0.0 \
    MAX_WORKERS=10 \
    ASYNC_TIMEOUT_SECONDS=30 \
    ENABLE_ADVANCED_STATS=true \
    ENABLE_WEATHER_DATA=true \
    ENABLE_INJURY_REPORTS=true

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

# Create directories dynamically based on ENV variables
RUN mkdir -p ${CACHE_DIR} $(dirname ${LOG_FILE}) && \
    chown -R appuser:appuser ${CACHE_DIR} $(dirname ${LOG_FILE})

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
