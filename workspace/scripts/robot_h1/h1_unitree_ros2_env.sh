#!/usr/bin/env bash

# Official Unitree ROS2 environment for H1 onboard PC.
# Keep conda active if you want, but ROS nodes must use /usr/bin/python3.

source /opt/ros/foxy/setup.bash
source /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# On this robot report eth0 = 192.168.123.162/24.
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
  <NetworkInterface name="eth0" priority="default" multicast="default" />
</Interfaces></General></Domain></CycloneDDS>'

export ROS_LOCALHOST_ONLY=0
