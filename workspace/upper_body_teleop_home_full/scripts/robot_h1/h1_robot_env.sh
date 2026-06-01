#!/usr/bin/env bash

cd ~/diploma/workspace/upper_body_teleop_home_full || return 1

source /opt/ros/humble/setup.bash

# Подключаем только реально существующие local_setup.
for p in \
  install/upper_body_msgs/share/upper_body_msgs/local_setup.bash \
  install/home_pipeline/share/home_pipeline/local_setup.bash \
  install/unitree_go/share/unitree_go/local_setup.bash \
  install/sdk2_h1_upper_body_bridge/share/sdk2_h1_upper_body_bridge/local_setup.bash
do
  if [ -f "$p" ]; then
    source "$p"
  fi
done

export PYTHONPATH="$PWD/src:$PWD/src/home_pipeline:$PYTHONPATH"

export UNITREE_NET_IFACE="${UNITREE_NET_IFACE:-enx00e04c36022c}"
export UNITREE_DOMAIN_ID="${UNITREE_DOMAIN_ID:-0}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-$UNITREE_DOMAIN_ID}"

# Если CycloneDDS установлен — используем его.
# Если нет — откатываемся на штатный FastDDS, чтобы ROS-ноды хотя бы запускались.
if ros2 pkg prefix rmw_cyclonedds_cpp >/dev/null 2>&1; then
  export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
else
  export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
fi

echo "[H1 ENV] UNITREE_NET_IFACE=$UNITREE_NET_IFACE"
echo "[H1 ENV] UNITREE_DOMAIN_ID=$UNITREE_DOMAIN_ID"
echo "[H1 ENV] ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "[H1 ENV] RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
