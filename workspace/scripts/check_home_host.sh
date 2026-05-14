#!/usr/bin/env bash
set -e

echo "===== OS ====="
uname -a || true
cat /etc/os-release || true

echo
echo "===== DOCKER ====="
docker --version || echo "docker: NOT FOUND"
docker compose version || echo "docker compose: NOT FOUND"
systemctl is-active docker || true
id -nG "$USER" | grep -qw docker && echo "user in docker group: YES" || echo "user in docker group: NO"

echo
echo "===== GUI ====="
echo "DISPLAY=${DISPLAY:-EMPTY}"
echo "XDG_SESSION_TYPE=${XDG_SESSION_TYPE:-EMPTY}"
xhost || true

echo
echo "===== CAMERA ====="
ls -la /dev/video* 2>/dev/null || echo "No /dev/video* devices found"
command -v v4l2-ctl >/dev/null && v4l2-ctl --list-devices || echo "v4l2-ctl not found on host; this is optional"

echo
echo "===== HINT ====="
echo "For mock test:   bash scripts/home_run_mock.sh"
echo "For camera test: bash scripts/home_run_camera.sh 0"
