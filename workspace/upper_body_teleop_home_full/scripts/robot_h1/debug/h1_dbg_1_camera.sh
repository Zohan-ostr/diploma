#!/usr/bin/env bash
set +e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/h1_dbg_common.sh"

CAMERA_ID="${1:-${CAMERA_ID:-0}}"

print_header "H1 DEBUG 1: CAMERA / MEDIAPIPE / POSE LANDMARKS"

cleanup() {
  echo
  log_note "Stopping camera debug wrapper"
  stop_monitor "camera"
  kill_pattern_everywhere "upper_body_teleop_runtime.webcam_mediapipe_to_udp"
  kill_pattern_everywhere "upper_body_teleop_runtime.udp_pose_landmarks_to_ros"
  ros_quick_snapshot "camera-exit"
  process_snapshot "camera-exit"
  tail_debug_logs
}
trap cleanup INT TERM EXIT

log_note "Starting camera stage with CAMERA_ID=$CAMERA_ID"

start_monitor "camera" 5

capture_bash "10_camera_start" "
cd '$PROJECT_DIR'
echo 'Launching real robot camera script...'
echo 'CAMERA_ID=$CAMERA_ID'
bash scripts/robot_h1/h1_1_start_camera.sh '$CAMERA_ID'
"

echo
echo "Camera start script returned. Keeping debug terminal alive for monitoring."
echo "Expected after successful start:"
echo "  /pose/landmarks Publisher count: 1"
echo
wait_forever "H1 DEBUG 1 CAMERA"
