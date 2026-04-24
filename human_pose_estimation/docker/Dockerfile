FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    unzip \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/human_pose_estimation

COPY docker/requirements-base.txt /tmp/requirements-base.txt
COPY docker/requirements-hpe.txt /tmp/requirements-hpe.txt
COPY docker/requirements-mmpose.txt /tmp/requirements-mmpose.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /tmp/requirements-base.txt && \
    python -m pip install -r /tmp/requirements-hpe.txt && \

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
