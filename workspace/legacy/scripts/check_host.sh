#!/usr/bin/env bash
set -euo pipefail

echo '===== OS ====='
uname -a
[ -f /etc/os-release ] && cat /etc/os-release

echo
echo '===== DOCKER ====='
command -v docker >/dev/null && docker --version || echo 'docker: NOT FOUND'
command -v docker >/dev/null && docker compose version || echo 'docker compose: NOT FOUND'
command -v systemctl >/dev/null && systemctl is-active docker || true
command -v systemctl >/dev/null && systemctl is-enabled docker || true
id | grep -q '\bdocker\b' && echo 'user in docker group: YES' || echo 'user in docker group: NO'

echo
echo '===== GPU ====='
command -v nvidia-smi >/dev/null && nvidia-smi || echo 'nvidia-smi: NOT FOUND'
command -v glxinfo >/dev/null && glxinfo -B || echo 'glxinfo: NOT FOUND (install mesa-utils if needed)'

echo
echo '===== X11 / WAYLAND ====='
echo "DISPLAY=${DISPLAY:-<empty>}"
echo "XDG_SESSION_TYPE=${XDG_SESSION_TYPE:-<empty>}"
echo "XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-<empty>}"

echo
echo '===== ROS / GAZEBO ON HOST (optional) ====='
command -v ros2 >/dev/null && ros2 --help >/dev/null && echo 'ros2: FOUND' || echo 'ros2: NOT FOUND'
command -v gazebo >/dev/null && gazebo --version || echo 'gazebo: NOT FOUND'

echo
echo '===== XHOST CHECK ====='
command -v xhost >/dev/null && xhost || echo 'xhost: NOT FOUND'

echo
echo '===== DONE ====='
