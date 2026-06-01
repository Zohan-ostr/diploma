FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
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
    iproute2 \
    patchelf \
    libosmesa6-dev \
    libgl1-mesa-dev \
    libglew-dev \
    libglfw3-dev \
    libboost-all-dev \
    libspdlog-dev \
    libyaml-cpp-dev \
    build-essential \
    ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# Official Unitree Python SDK2: ChannelPublisher/Subscriber, IDL messages, CRC
RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python.git /opt/unitree_sdk2_python \
    && python3 -m pip install --no-cache-dir -e /opt/unitree_sdk2_python

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

# ===== Unitree SDK2 for unitree_mujoco =====
RUN mkdir -p /opt/unitree_src && \
    cd /opt/unitree_src && \
    git clone https://github.com/unitreerobotics/unitree_sdk2.git && \
    cd unitree_sdk2 && \
    rm -rf build && mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/opt/unitree_robotics && \
    make -j$(nproc) && \
    make install && \
    rm -rf /opt/unitree_src/unitree_sdk2/build

ENV CMAKE_PREFIX_PATH=/opt/unitree_robotics:${CMAKE_PREFIX_PATH}
ENV LD_LIBRARY_PATH=/opt/unitree_robotics/lib:${LD_LIBRARY_PATH}

# ===== Python MuJoCo for tests/tools =====
RUN python3 -m pip install --no-cache-dir mujoco pygame

# ===== MuJoCo runtime path for unitree_mujoco C++ simulator =====
ENV LD_LIBRARY_PATH=/workspace/external/unitree_mujoco/simulate/mujoco/lib:${LD_LIBRARY_PATH}

# Added for RGB-D RealSense camera support
RUN python3 -m pip install --no-cache-dir pyrealsense2
