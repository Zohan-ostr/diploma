FROM osrf/ros:jazzy-desktop

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y \
    git \
    wget \
    cmake \
    build-essential \
    python3-pip \
    python3-dev \
    python3-numpy \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-opencv \
    v4l-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgtk2.0-dev \
    mesa-utils \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    && rm -rf /var/lib/apt/lists/*

# MediaPipe on ARM64/Jetson can depend on wheel availability.
# Keep versions pinned where possible, but do not fail the whole image if mediapipe wheel is unavailable.
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir "numpy<2" "opencv-python>=4.9,<5" "matplotlib>=3.8,<4" "mujoco>=3.2,<4" && \
    (python3 -m pip install --no-cache-dir "mediapipe==0.10.14" || \
     echo "WARNING: mediapipe wheel was not installed. Install an ARM64-compatible MediaPipe package on Jetson if needed.")

# Official Unitree MuJoCo repository. The G1 model is expected at:
# /opt/unitree_mujoco/unitree_robots/g1/g1_29dof.xml
RUN git clone --depth 1 https://github.com/unitreerobotics/unitree_mujoco.git /opt/unitree_mujoco || true

ENV UNITREE_MUJOCO_ROOT=/opt/unitree_mujoco
ENV G1_MUJOCO_XML=/opt/unitree_mujoco/unitree_robots/g1/g1_29dof.xml

WORKDIR /workspace
CMD ["bash"]
