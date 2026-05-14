#!/usr/bin/env bash
set -e

cd /workspace

export MUJOCO_NET_IFACE="${MUJOCO_NET_IFACE:-wlp0s20f3}"

bash /workspace/scripts/sim_configure_mujoco_h1.sh

cd /workspace/external/unitree_mujoco/simulate/build

# Важно:
# unitree_mujoco использует unitree_sdk2 и CycloneDDS.
# Нельзя запускать его с ROS Humble LD_LIBRARY_PATH,
# иначе он может подхватить /opt/ros/humble/lib/.../libddsc.so.0
# и упасть с free()/malloc() corruption.
MUJOCO_LIB="/workspace/external/unitree_mujoco/simulate/mujoco/lib"
UNITREE_LIB="/opt/unitree_robotics/lib"

echo "Running unitree_mujoco with clean DDS runtime"
echo "MUJOCO_NET_IFACE=$MUJOCO_NET_IFACE"
echo "LD_LIBRARY_PATH=$UNITREE_LIB:$MUJOCO_LIB:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"

env \
  -u AMENT_PREFIX_PATH \
  -u COLCON_PREFIX_PATH \
  -u ROS_DISTRO \
  -u ROS_VERSION \
  -u ROS_PYTHON_VERSION \
  -u RMW_IMPLEMENTATION \
  LD_LIBRARY_PATH="$UNITREE_LIB:$MUJOCO_LIB:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu" \
  ./unitree_mujoco
