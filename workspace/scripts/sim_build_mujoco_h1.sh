#!/usr/bin/env bash
set -e

export LD_LIBRARY_PATH=/workspace/external/unitree_mujoco/simulate/mujoco/lib:${LD_LIBRARY_PATH}

cd /workspace/external/unitree_mujoco/simulate

echo "Checking MuJoCo symlink..."
ls -lah mujoco
find mujoco -name "glfw_adapter.h" -o -name "libmujoco.so"

echo
echo "Building unitree_mujoco..."
rm -rf build
mkdir -p build
cd build

cmake .. -DCMAKE_PREFIX_PATH=/opt/unitree_robotics
make -j"$(nproc)"

echo
echo "Executables:"
find . -maxdepth 2 -type f -executable -print
