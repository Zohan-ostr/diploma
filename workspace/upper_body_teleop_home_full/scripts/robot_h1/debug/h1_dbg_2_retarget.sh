#!/usr/bin/env bash
set +e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/h1_dbg_common.sh"

print_header "H1 DEBUG 2: RETARGET"

cleanup() {
  echo
  log_note "Stopping retarget debug wrapper"
  stop_monitor "retarget"
  kill_pattern_everywhere "upper_body_teleop_runtime.vector_fabrik_retarget"
  ros_quick_snapshot "retarget-exit"
  process_snapshot "retarget-exit"
  tail_debug_logs
}
trap cleanup INT TERM EXIT

log_note "Starting retarget stage"

start_monitor "retarget" 5

ros_quick_snapshot "before-retarget"

capture_bash "20_retarget_start" "
cd '$PROJECT_DIR'
echo 'Launching retarget script...'
bash scripts/robot_h1/h1_2_start_retarget.sh
"

sleep 3
ros_quick_snapshot "after-retarget-start"
process_snapshot "after-retarget-start"

echo
echo "Retarget start script returned. Keeping debug terminal alive for monitoring."
echo "Expected:"
echo "  /pose/landmarks Subscription count: 1"
echo "  /upper_body/command_geom Publisher count: 1"
echo
wait_forever "H1 DEBUG 2 RETARGET"
