#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo " SIM 3: START RETARGET FOR SIMULATION"
echo "============================================================"
echo "Using: two-stage segmented FK search for pitch/roll"
echo "Output: /upper_body/command_geom"
echo "NOTE: direct SDK2 retarget is only for real H1 robot"
echo "============================================================"

exec bash scripts/simulation/sim_3_start_projection_retarget.sh "$@"
