#!/usr/bin/env bash
set +e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/h1_dbg_common.sh"

print_header "H1 DEBUG 0: PREFLIGHT / NO MOTION"

log_note "Starting preflight. This stage must not move the robot."

capture_bash "00_env" "
cd '$PROJECT_DIR'
echo '=== date ==='
date
echo
echo '=== pwd ==='
pwd
echo
echo '=== git status ==='
git status --short 2>/dev/null || true
echo
echo '=== important files ==='
for f in \
  scripts/robot_h1/h1_0_check_sdk2_callback.sh \
  scripts/robot_h1/h1_1_start_camera.sh \
  scripts/robot_h1/h1_2_start_retarget.sh \
  scripts/robot_h1/h1_4_start_sdk2_sender.sh \
  scripts/robot_h1/h1_5_start_calibration.sh \
  scripts/robot_h1/h1_6_check.sh \
  src/upper_body_teleop_runtime/vector_fabrik_retarget.py \
  src/upper_body_teleop_runtime/webcam_mediapipe_to_udp.py \
  src/upper_body_teleop_runtime/udp_pose_landmarks_to_ros.py \
  src/upper_body_teleop_runtime/h1_sdk2py_upper_body_sender.py
do
  if [ -f \"\$f\" ]; then
    echo \"OK: \$f\"
  else
    echo \"MISSING: \$f\"
  fi
done
"

network_snapshot "preflight"
process_snapshot "preflight"

capture_bash "01_python_compile" "
cd '$PROJECT_DIR'
PYTHONPATH=\"\$PWD/src:\$PWD/src/home_pipeline:\$PWD/src/h1_robot_adapter:\$PYTHONPATH\" \
python3 -m py_compile \
  src/upper_body_teleop_runtime/vector_fabrik_retarget.py \
  src/upper_body_teleop_runtime/webcam_mediapipe_to_udp.py \
  src/upper_body_teleop_runtime/udp_pose_landmarks_to_ros.py \
  src/upper_body_teleop_runtime/h1_sdk2py_upper_body_sender.py
"

capture_bash "02_docker_state" "
cd '$PROJECT_DIR'
echo '=== docker version ==='
docker --version || true
docker compose version || true
echo
echo '=== docker ps ==='
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' || true
"

if [ -n "$UNITREE_NET_IFACE" ]; then
  capture_bash "03_sdk2_lowstate_check" "
cd '$PROJECT_DIR'
echo 'Checking SDK2 lowstate callback for 25 sec...'
timeout --foreground 25s bash -lc '
  export UNITREE_NET_IFACE=\"$UNITREE_NET_IFACE\"
  export UNITREE_DOMAIN_ID=\"$UNITREE_DOMAIN_ID\"
  bash scripts/robot_h1/h1_0_check_sdk2_callback.sh
'
echo 'SDK2 lowstate check finished with rc='$?
"
else
  log_note "UNITREE_NET_IFACE is not set, SDK2 lowstate check skipped."
  echo
  echo "WARNING: UNITREE_NET_IFACE is not set."
  echo "Перед реальным запуском задай интерфейс, например:"
  echo "  export UNITREE_NET_IFACE=enx00e04c36022c"
fi

ros_quick_snapshot "preflight-final"
process_snapshot "preflight-final"
network_snapshot "preflight-final"

echo
echo "============================================================"
echo " PREFLIGHT DONE"
echo "============================================================"
echo "Проверь логи:"
echo "  $LOG_ROOT"
echo
echo "Дальше запускай терминалы 1–5."
echo "============================================================"

tail_debug_logs
