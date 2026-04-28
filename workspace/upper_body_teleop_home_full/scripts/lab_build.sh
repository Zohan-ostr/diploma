#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
docker compose -f compose/compose.lab.yaml build
