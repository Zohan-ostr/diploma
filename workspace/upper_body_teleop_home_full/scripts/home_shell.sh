#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
xhost +local:docker >/dev/null || true
docker compose -f compose.home.yaml run --rm home-dev bash
