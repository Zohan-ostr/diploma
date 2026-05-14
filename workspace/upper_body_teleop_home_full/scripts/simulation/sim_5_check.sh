#!/usr/bin/env bash
set -euo pipefail

echo "sim_5_check.sh переименован."
echo "Теперь проверка запускается так:"
echo "  bash scripts/simulation/sim_6_check.sh"
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/sim_6_check.sh"
