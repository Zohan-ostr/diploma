#!/usr/bin/env bash
set -euo pipefail

cd "${HOME}/WS_OZ"

if [ ! -d unitree_sdk2 ]; then
  echo "Cloning unitree_sdk2..."
  git clone https://github.com/unitreerobotics/unitree_sdk2.git
else
  echo "unitree_sdk2 already exists"
fi

sudo apt update
sudo apt install -y cmake g++ build-essential libyaml-cpp-dev libeigen3-dev libboost-all-dev libspdlog-dev libfmt-dev

cd "${HOME}/WS_OZ/unitree_sdk2"

# Disable problematic G1 dual arm example if it causes yaml-cpp link errors on Ubuntu 20.04.
if [ -f example/g1/CMakeLists.txt ]; then
  python3 - <<'PY'
from pathlib import Path
p = Path('example/g1/CMakeLists.txt')
text = p.read_text(encoding='utf-8', errors='ignore')
out = []
for line in text.splitlines():
    if 'g1_dual_arm_example' in line and not line.lstrip().startswith('#'):
        out.append('# DISABLED_FOR_H1_TEST: ' + line)
    else:
        out.append(line)
p.write_text('\n'.join(out) + '\n', encoding='utf-8')
print('patched optional G1 example:', p)
PY
fi

sudo rm -rf build
mkdir build
cd build
cmake ..
make -j"$(nproc)"
sudo make install

echo "===== SDK2 install check ====="
ls /usr/local/include/unitree
ls /usr/local/lib | grep -i unitree || true
