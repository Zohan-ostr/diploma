#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
rm -rf build
mkdir build
cd build
cmake ..
make -j"$(nproc)"
