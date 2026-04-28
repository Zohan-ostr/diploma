FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-opencv \
    python3-dev \
    python3-numpy \
    v4l-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgtk2.0-dev \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Важно: фиксируем старый стабильный стек под MediaPipe Solutions API
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip uninstall -y mediapipe mediapipe-numpy2 numpy opencv-python matplotlib || true && \
    python3 -m pip install --no-cache-dir \
        "numpy==1.26.4" \
        "opencv-python==4.9.0.80" \
        "matplotlib==3.8.4" \
        "mediapipe==0.10.14"

WORKDIR /workspace

CMD ["bash"]
