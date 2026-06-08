#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."
source scripts/robot_h1/h1_env.sh

print_h1_env

echo "============================================================"
echo " H1 6: CHECK REAL H1 TOPICS"
echo "============================================================"

echo
echo "=== NODES ==="
ros2 node list || true

echo
echo "=== TOPICS ==="
ros2 topic list -t || true

echo
echo "=== /pose/landmarks ==="
ros2 topic info /pose/landmarks -v || true

echo
echo "=== /upper_body/start_calibration ==="
ros2 topic info /upper_body/start_calibration -v || true

echo
echo "=== /upper_body/command_geom ==="
ros2 topic info /upper_body/command_geom -v || true

echo
echo "=== one /upper_body/command_geom sample ==="
timeout 5 ros2 topic echo /upper_body/command_geom --once || true
