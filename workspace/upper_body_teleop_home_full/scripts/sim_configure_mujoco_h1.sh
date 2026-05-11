#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$PROJECT_ROOT/external/unitree_mujoco/simulate/config.yaml"

IFACE="${MUJOCO_NET_IFACE:-wlp0s20f3}"

cat > "$CONFIG" <<CFG
robot: "h1"
robot_scene: "scene.xml"

domain_id: 42
interface: "$IFACE"

use_joystick: 0
joystick_type: "xbox"
joystick_device: "/dev/input/js0"
joystick_bits: 16

print_scene_information: 0

enable_elastic_band: 1
CFG

echo "Wrote config: $CONFIG"
cat "$CONFIG"
