#!/bin/bash
set -e

source /opt/ros/humble/setup.bash

if [ -f /workspaces/h1_teleop_ws/install/setup.bash ]; then
  source /workspaces/h1_teleop_ws/install/setup.bash
fi

exec "$@"
