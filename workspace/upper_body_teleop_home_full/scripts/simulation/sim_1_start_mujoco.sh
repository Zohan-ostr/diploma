#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo " SIM 1: START H1 MUJOCO SIMULATION"
echo "============================================================"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "============================================================"

xhost +local:docker >/dev/null 2>&1 || true

bash scripts/host_rebuild_and_run_mujoco_h1.sh
