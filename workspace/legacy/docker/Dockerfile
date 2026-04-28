FROM osrf/ros:humble-desktop

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV WS=/workspaces/h1_teleop_ws

RUN apt-get update && apt-get install -y --no-install-recommends \
    gazebo \
    libgazebo-dev \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-plugins \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-rviz2 \
    ros-humble-tf2-tools \
    ros-humble-xacro \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-pip \
    git \
    curl \
    nano \
    vim \
    mesa-utils \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init 2>/dev/null || true && rosdep update

RUN mkdir -p ${WS}/src
WORKDIR ${WS}

COPY docker/entrypoint.sh /ros_entrypoint_custom.sh
RUN chmod +x /ros_entrypoint_custom.sh

ENTRYPOINT ["/ros_entrypoint_custom.sh"]
CMD ["bash"]
