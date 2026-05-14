#!/usr/bin/env bash
set -euo pipefail

echo "sim_2_start_bridge.sh переименован."
echo "Теперь bridge запускается так:"
echo "  bash scripts/simulation/sim_3_start_bridge.sh"
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/sim_3_start_bridge.sh"
