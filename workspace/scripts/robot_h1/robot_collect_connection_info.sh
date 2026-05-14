#!/usr/bin/env bash
set -euo pipefail

REPORT=${REPORT:-/tmp/h1_robot_connection_report.txt}
{
  echo "============================================================"
  echo " H1 ROBOT CONNECTION REPORT"
  echo "============================================================"
  echo "date:      $(date)"
  echo "hostname:  $(hostname)"
  echo "user:      $(whoami)"
  echo "pwd:       $(pwd)"
  echo "report:    $REPORT"
  echo "============================================================"
  echo

  echo "===== uname -a ====="
  uname -a || true
  echo

  echo "===== lsb_release -a ====="
  lsb_release -a 2>/dev/null || true
  echo

  echo "===== ip -br addr ====="
  ip -br addr || true
  echo

  echo "===== ip route ====="
  ip route || true
  echo

  echo "===== ROS 2 CHECK ====="
  if [ -f /opt/ros/foxy/setup.bash ]; then
    echo "ROS Foxy found: /opt/ros/foxy"
  else
    echo "ROS Foxy not found"
  fi
  if [ -f /opt/ros/humble/setup.bash ]; then
    echo "ROS Humble found: /opt/ros/humble"
  else
    echo "ROS Humble not found"
  fi
  echo

  echo "===== ros2 topic list -t ====="
  source /opt/ros/foxy/setup.bash 2>/dev/null || true
  source /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash 2>/dev/null || true
  export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}
  export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
  export ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-0}
  timeout 3 ros2 topic list -t 2>/dev/null || true
  echo

  echo "===== PYTHON / UNITREE SDK CHECK ====="
  /usr/bin/python3 --version || true
  /usr/bin/python3 - <<'PY' || true
try:
    import rclpy
    print('OK import rclpy')
except Exception as e:
    print('FAIL import rclpy:', e)
try:
    import unitree_sdk2py
    print('OK import unitree_sdk2py')
except Exception as e:
    print('FAIL import unitree_sdk2py:', e)
PY
  echo

  echo "===== UNITREE SDK2 INSTALL CHECK ====="
  ls /usr/local/include/unitree 2>/dev/null || true
  ls /usr/local/lib | grep -i unitree 2>/dev/null || true
  find "$HOME/WS_OZ/unitree_sdk2" -name dds.hpp -o -name libddsc.so -o -name libddscxx.so 2>/dev/null || true
  echo

  echo "===== DETECT NETWORK INTERFACES ====="
  echo -n "candidate interfaces: "
  ip -br addr | awk '$1 != "lo" {print $1}' | tr '\n' ' '
  echo
  echo "route to default:"
  ip route get 8.8.8.8 2>/dev/null || true
  echo "route to robot subnet peer example 192.168.123.1:"
  ip route get 192.168.123.1 2>/dev/null || true
  echo

  echo "===== SUMMARY ====="
  echo "Expected robot iface usually: eth0"
  echo "Expected robot IP in this setup: 192.168.123.162"
  echo "Expected laptop IP in this setup: 192.168.123.200"
  echo "Report saved to: $REPORT"
  echo "============================================================"
} | tee "$REPORT"
