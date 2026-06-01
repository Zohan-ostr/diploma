#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SETUP EXTERNAL REPOSITORIES
# ============================================================
#
# Назначение:
#   скачать внешние репозитории проекта в external/.
#
# Python-библиотеки НЕ устанавливаются здесь.
# Они устанавливаются внутри Docker-контейнера через Dockerfile.
# ============================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p external

echo "============================================================"
echo " SETUP EXTERNAL REPOSITORIES"
echo "============================================================"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "TARGET:      $PROJECT_DIR/external"
echo "============================================================"

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git not found"
  echo "Install git first:"
  echo "  sudo apt update && sudo apt install -y git"
  exit 1
fi

clone_if_missing() {
  local url="$1"
  local dst="$2"

  if [ -d "$dst" ]; then
    echo "[SKIP] $dst already exists"
    return 0
  fi

  echo "[CLONE] $url -> $dst"
  git clone "$url" "$dst"
}

clone_if_missing "https://github.com/unitreerobotics/unitree_ros2.git" external/unitree_ros2
clone_if_missing "https://github.com/unitreerobotics/unitree_sdk2.git" external/unitree_sdk2
clone_if_missing "https://github.com/unitreerobotics/unitree_sdk2_python.git" external/unitree_sdk2_python
clone_if_missing "https://github.com/unitreerobotics/unitree_mujoco.git" external/unitree_mujoco

mkdir -p external/mujoco_download

echo
echo "============================================================"
echo " EXTERNAL REPOSITORIES READY"
echo "============================================================"
echo "Next step:"
echo "  docker compose -f compose/compose.home.yaml build home-dev"
