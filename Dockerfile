# TIER 1 Rejuvenation Suite - Multi-stage Dockerfile
# =================================================
#
# This Dockerfile creates a complete environment for the TIER 1 Cellular
# Rejuvenation Suite with all system dependencies, optimized for both
# development and production use.
#
# Build stages:
#   1. base - System dependencies and conda
#   2. builder - Python environment and dependencies
#   3. production - Final optimized image
#
# Usage:
#   docker build -t tier1-rejuvenation-suite .
#   docker run -it tier1-rejuvenation-suite tier1 --help

# =============================================================================
# Stage 1: Base system with conda and system dependencies
# =============================================================================
FROM ubuntu:22.04 as base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    gfortran \
    make \
    cmake \
    pkg-config \
    \
    # OpenMP and parallel computing
    libomp-dev \
    libgomp1 \
    \
    # HDF5 and scientific libraries
    libhdf5-dev \
    libhdf5-serial-dev \
    hdf5-tools \
    libhdf5-103 \
    \
    # BLAS/LAPACK optimized libraries
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    libatlas-base-dev \
    \
    # Additional scientific libraries
    libfftw3-dev \
    libgsl-dev \
    \
    # Graphics and visualization
    libcairo2-dev \
    libpango1.0-dev \
    libgdk-pixbuf2.0-dev \
    libgtk-3-dev \
    \
    # Network and utilities
    curl \
    wget \
    git \
    unzip \
    vim \
    nano \
    htop \
    \
    # Python development
    python3-dev \
    python3-pip \
    python3-venv \
    \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && conda clean --all -y \
    && conda config --set auto_update_conda false

# Configure conda
RUN conda config --add channels conda-forge \
    && conda config --add channels bioconda \
    && conda config --add channels pytorch \
    && conda config --set channel_priority strict

# =============================================================================
# Stage 2: Builder - Install Python dependencies
# =============================================================================
FROM base as builder

# Set working directory
WORKDIR /app

# Copy environment files
COPY environment-prod.yml .
COPY requirements.txt .
COPY pyproject.toml .
COPY setup.py .

# Create conda environment from file
RUN conda env create -f environment-prod.yml \
    && conda clean --all -y

# Activate environment and install additional pip packages
ENV CONDA_DEFAULT_ENV=tier1-rejuvenation-suite-prod
ENV PATH=/opt/conda/envs/tier1-rejuvenation-suite-prod/bin:$PATH

# Install additional pip packages not available in conda
RUN pip install --no-cache-dir \
    scrublet>=0.2.3 \
    harmonypy>=0.0.9 \
    scanorama>=1.7.3 \
    bbknn>=1.5.1 \
    adjusttext>=0.7.3 \
    session-info>=1.0.0

# Copy source code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# =============================================================================
# Stage 3: Production - Final optimized image
# =============================================================================
FROM base as production

# Create non-root user
RUN useradd -m -s /bin/bash tier1user

# Set working directory
WORKDIR /app

# Copy conda environment from builder
COPY --from=builder /opt/conda/envs/tier1-rejuvenation-suite-prod /opt/conda/envs/tier1-rejuvenation-suite-prod

# Copy application code
COPY --from=builder /app .

# Set up environment
ENV CONDA_DEFAULT_ENV=tier1-rejuvenation-suite-prod
ENV PATH=/opt/conda/envs/tier1-rejuvenation-suite-prod/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Create directories for data and results
RUN mkdir -p /app/data /app/results /app/models \
    && chown -R tier1user:tier1user /app

# Switch to non-root user
USER tier1user

# Verify installation
RUN python -c "import tier1_suite; print('TIER 1 Suite installed successfully')" \
    && tier1 --help

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD tier1 version || exit 1

# Default command
CMD ["tier1", "--help"]

# =============================================================================
# Development stage (optional) - Includes dev tools
# =============================================================================
FROM builder as development

# Install development dependencies
COPY requirements-dev.in .
RUN pip install --no-cache-dir \
    pytest>=7.0.0 \
    pytest-cov>=4.0.0 \
    black>=22.0.0 \
    flake8>=5.0.0 \
    mypy>=0.991 \
    jupyterlab>=3.4.0 \
    notebook>=6.4.0

# Expose Jupyter port
EXPOSE 8888

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# =============================================================================
# Labels and metadata
# =============================================================================
LABEL maintainer="Kemal Yaylali <kemal.yaylali@gmail.com>"
LABEL version="1.0.0"
LABEL description="TIER 1 Cellular Rejuvenation Suite - Biologically validated analysis tools"
LABEL org.opencontainers.image.source="https://github.com/lynchaos/tier1-rejuvenation-suite"
LABEL org.opencontainers.image.documentation="https://github.com/lynchaos/tier1-rejuvenation-suite#readme"
LABEL org.opencontainers.image.licenses="MIT"