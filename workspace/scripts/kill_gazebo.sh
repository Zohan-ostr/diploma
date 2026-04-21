#!/usr/bin/env bash
set -euo pipefail
pkill -9 -f gzserver || true
pkill -9 -f gzclient || true
pkill -9 -f 'gazebo --verbose' || true
sleep 2
ps aux | grep -E 'gzserver|gzclient|gazebo' | grep -v grep || true
