# Dockerfile -- container image for the integrated CV system server.
#
# Build:  docker build -t integrated-cv-system .
# Run:    docker run --gpus all -p 8000:8000 integrated-cv-system
#
# Notes:
# - Start from a CUDA runtime base so the GPU is available inside the
#   container (the host needs the NVIDIA Container Toolkit).
# - Install Python deps first, as their own layer, so code changes do
#   not invalidate the (slow) dependency layer in the build cache.
# - Run as a non-root user; expose the port; use a health check.

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# System packages: Python and the shared libs OpenCV/aiortc need.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        libgl1 libglib2.0-0 libsm6 libxext6 \
        libavdevice-dev libopus0 libvpx7 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer (cached unless requirements.txt changes).
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Application code.
COPY server/ server/
COPY web/ web/
COPY scripts/ scripts/
COPY models/ models/

# Non-root user.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python3 -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# Uvicorn serves the FastAPI app; bind to all interfaces in-container.
CMD ["python3", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
