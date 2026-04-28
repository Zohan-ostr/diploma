#!/usr/bin/env bash
set -e

echo "===== OS ====="
uname -a
cat /etc/os-release || true

echo
echo "===== ARCH ====="
uname -m

echo
echo "===== DOCKER ====="
docker --version || true
docker compose version || true
systemctl is-active docker || true
systemctl is-enabled docker || true
groups | grep -qw docker && echo "user in docker group: YES" || echo "user in docker group: NO"

echo
echo "===== NVIDIA / JETSON ====="
which nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi: NOT FOUND or not supported on Jetson"
[ -f /etc/nv_tegra_release ] && cat /etc/nv_tegra_release || true

echo
echo "===== CAMERA ====="
ls -l /dev/video* 2>/dev/null || echo "No /dev/video* found"
groups | grep -qw video && echo "user in video group: YES" || echo "user in video group: NO"

echo
echo "===== GUI ====="
echo "DISPLAY=${DISPLAY:-}"
echo "XDG_SESSION_TYPE=${XDG_SESSION_TYPE:-}"
xhost 2>/dev/null | head -n 5 || true

echo
echo "===== OPTIONAL HOST ROS/MUJOCO ====="
which ros2 >/dev/null 2>&1 && ros2 --version || echo "ros2: not found on host (OK, container has ROS)"
python3 - <<'PY' || true
try:
    import mujoco
    print('host mujoco:', mujoco.__version__)
except Exception as e:
    print('host mujoco: not importable:', e)
PY
