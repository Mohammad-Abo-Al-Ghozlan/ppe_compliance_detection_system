# Stage 1: Build & Install dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies required for building python packages and cv2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt


# Stage 2: Runtime (supports CPU and GPU if nvidia-container-runtime is present)
FROM python:3.11-slim

WORKDIR /app

# Install runtime system dependencies for cv2 and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder and install
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache /wheels/*

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Expose API port
EXPOSE 8000

# Default command to run the API (assumes best_model.pt is mounted or present)
# Using python -m uvicorn to ensure it runs correctly from module path
CMD ["python", "api.py", "--config", "configs/config.yaml", "--model", "best_model.pt", "--port", "8000"]
