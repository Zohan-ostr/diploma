#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-h1_mujoco_sim}"

ROS_DOMAIN_ID_VALUE="${ROS_DOMAIN_ID_VALUE:-42}"

INPUT_TOPIC="${INPUT_TOPIC:-/pose/landmarks}"
OUTPUT_TOPIC="${OUTPUT_TOPIC:-/upper_body/command_geom}"

CALIBRATION_FRAMES="${CALIBRATION_FRAMES:-45}"

MAP_X_FROM_Z="${MAP_X_FROM_Z:-0.0}"

LANDMARK_ALPHA="${LANDMARK_ALPHA:-0.25}"
JOINT_ALPHA="${JOINT_ALPHA:-0.20}"
MAX_JOINT_STEP="${MAX_JOINT_STEP:-0.060}"

PITCH_GAIN="${PITCH_GAIN:-1.25}"
ROLL_GAIN="${ROLL_GAIN:-1.20}"
ELBOW_GAIN="${ELBOW_GAIN:-1.15}"

LEFT_YAW_DOWN="${LEFT_YAW_DOWN:-3.0}"
RIGHT_YAW_DOWN="${RIGHT_YAW_DOWN:--3.0}"

LEFT_PITCH_SIGN="${LEFT_PITCH_SIGN:--1.0}"
RIGHT_PITCH_SIGN="${RIGHT_PITCH_SIGN:--1.0}"
LEFT_ROLL_SIGN="${LEFT_ROLL_SIGN:-1.0}"
RIGHT_ROLL_SIGN="${RIGHT_ROLL_SIGN:--1.0}"
LEFT_ELBOW_SIGN="${LEFT_ELBOW_SIGN:--1.0}"
RIGHT_ELBOW_SIGN="${RIGHT_ELBOW_SIGN:--1.0}"

CID="$(docker ps --filter "name=^/${CONTAINER_NAME}$" --format "{{.ID}}" | head -n 1)"

if [ -z "$CID" ]; then
  echo "ERROR: running container '$CONTAINER_NAME' not found."
  docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"
  exit 1
fi

echo "Found container:"
docker ps --filter "id=$CID" --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"

echo "Killing old geometric retarget..."
docker exec "$CID" bash -lc "pkill -f '[h]1_geometric_retarget_node' || true"

echo
echo "Starting clean geometric retarget:"
echo "  INPUT_TOPIC:       $INPUT_TOPIC"
echo "  OUTPUT_TOPIC:      $OUTPUT_TOPIC"
echo "  MAP_X_FROM_Z:      $MAP_X_FROM_Z"
echo "  PITCH_GAIN:        $PITCH_GAIN"
echo "  ROLL_GAIN:         $ROLL_GAIN"
echo "  ELBOW_GAIN:        $ELBOW_GAIN"
echo "  LEFT_YAW_DOWN:     $LEFT_YAW_DOWN"
echo "  RIGHT_YAW_DOWN:    $RIGHT_YAW_DOWN"
echo "  SIGNS pitch L/R:   $LEFT_PITCH_SIGN / $RIGHT_PITCH_SIGN"
echo "  SIGNS roll  L/R:   $LEFT_ROLL_SIGN / $RIGHT_ROLL_SIGN"
echo "  SIGNS elbow L/R:   $LEFT_ELBOW_SIGN / $RIGHT_ELBOW_SIGN"
echo

docker exec -it "$CID" bash -lc "
  set -e

  cd /workspace

  source /opt/ros/humble/setup.bash
  source /workspace/install/setup.bash 2>/dev/null || true

  export ROS_DOMAIN_ID='$ROS_DOMAIN_ID_VALUE'
  export ROS_LOCALHOST_ONLY=0

  colcon build --symlink-install --packages-select h1_robot_adapter
  source /workspace/install/setup.bash

  ros2 run h1_robot_adapter h1_geometric_retarget_node --ros-args \
    -p input_topic:='$INPUT_TOPIC' \
    -p output_topic:='$OUTPUT_TOPIC' \
    -p calibration_frames:='$CALIBRATION_FRAMES' \
    -p map_x_from_z:='$MAP_X_FROM_Z' \
    -p landmark_alpha:='$LANDMARK_ALPHA' \
    -p joint_alpha:='$JOINT_ALPHA' \
    -p max_joint_step:='$MAX_JOINT_STEP' \
    -p pitch_gain:='$PITCH_GAIN' \
    -p roll_gain:='$ROLL_GAIN' \
    -p elbow_gain:='$ELBOW_GAIN' \
    -p left_yaw_down:='$LEFT_YAW_DOWN' \
    -p right_yaw_down:='$RIGHT_YAW_DOWN' \
    -p left_pitch_sign:='$LEFT_PITCH_SIGN' \
    -p right_pitch_sign:='$RIGHT_PITCH_SIGN' \
    -p left_roll_sign:='$LEFT_ROLL_SIGN' \
    -p right_roll_sign:='$RIGHT_ROLL_SIGN' \
    -p left_elbow_sign:='$LEFT_ELBOW_SIGN' \
    -p right_elbow_sign:='$RIGHT_ELBOW_SIGN'
"
