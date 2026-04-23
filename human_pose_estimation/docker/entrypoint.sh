#!/usr/bin/env bash
set -e

echo "[entrypoint] working directory: $(pwd)"
echo "[entrypoint] python: $(python --version 2>/dev/null || true)"

exec "$@"
