# Use an official lightweight Python 3.11 image (PaddlePaddle 3.x is fully supported on 3.11)
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLAGS_use_mkldnn=0 \
    PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=0 \
    PORT=8000

# Install Linux system dependencies needed for:
# - OpenCV GUI/Rendering fallback (libgl1, libglib2.0-0, libsm6, libxext6)
# - pyzbar barcode scanner (libzbar0)
# - PaddlePaddle / OpenMP (libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libzbar0 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY . .

# Render exposes the application port via the PORT environment variable
EXPOSE ${PORT}

# Run Uvicorn dynamically binding to the port assigned by Render (defaults to 8000 locally)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
