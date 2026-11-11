FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install PyTorch CPU-only
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install torch==2.8.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements, excluding torch and torch-geometric
COPY requirements.txt /tmp/requirements.txt

RUN grep -Eiv '^[[:space:]]*(torch|torch-geometric)([<=>!~].*)?$' \
    /tmp/requirements.txt > /tmp/requirements-cpu.txt && \
    pip install -r /tmp/requirements-cpu.txt

# Install PyTorch Geometric
RUN pip install torch-geometric==2.6.1

# JupyterLab
RUN pip install jupyterlab ipykernel ipywidgets

# Copy entire project
COPY . /workspace

# Persistent output directory
RUN mkdir -p /workspace/logs

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--ServerApp.root_dir=/workspace"]