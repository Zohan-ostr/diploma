#!/usr/bin/env bash
set -e
ros2 bag record /pose/landmarks /upper_body/command /joint_states /tf /tf_static
