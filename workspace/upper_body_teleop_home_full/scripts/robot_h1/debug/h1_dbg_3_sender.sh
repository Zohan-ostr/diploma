#!/usr/bin/env bash
set +e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/h1_dbg_common.sh"

print_header "H1 DEBUG 3: REAL H1 SDK2 SENDER"

cleanup() {
  echo
  log_note "Stopping sender debug wrapper"
  stop_monitor "sender"
  kill_pattern_everywhere "upper_body_teleop_runtime.h1_sdk2py_upper_body_sender"
  ros_quick_snapshot "sender-exit"
  process_snapshot "sender-exit"
  network_snapshot "sender-exit"
  tail_debug_logs
}
trap cleanup INT TERM EXIT

if [ -z "$UNITREE_NET_IFACE" ]; then
  echo "ERROR: UNITREE_NET_IFACE is not set."
  echo "Пример:"
  echo "  export UNITREE_NET_IFACE=enx00e04c36022c"
  exit 1
fi

echo
echo "ВНИМАНИЕ: это этап реального управления H1."
echo "Перед продолжением проверь:"
echo "  1) робот стоит устойчиво;"
echo "  2) рядом с руками нет людей и препятствий;"
echo "  3) есть возможность быстро остановить управление;"
echo "  4) rt/lowstate проверен на этапе preflight;"
echo "  5) PD для реального H1 не менялись: kp=60.0, kd=1.5."
echo
read -r -p "Для запуска sender введи START: " ANSWER

if [ "$ANSWER" != "START" ]; then
  echo "Sender launch cancelled."
  exit 0
fi

log_note "Starting real H1 sender stage"

network_snapshot "before-sender"
ros_quick_snapshot "before-sender"
process_snapshot "before-sender"

capture_bash "30_sdk2_lowstate_before_sender" "
cd '$PROJECT_DIR'
echo 'Short SDK2 lowstate check before sender...'
timeout --foreground 15s bash -lc '
  export UNITREE_NET_IFACE=\"$UNITREE_NET_IFACE\"
  export UNITREE_DOMAIN_ID=\"$UNITREE_DOMAIN_ID\"
  bash scripts/robot_h1/h1_0_check_sdk2_callback.sh
'
echo 'lowstate check rc='$?
"

start_monitor "sender" 5

capture_bash "31_sender_start" "
cd '$PROJECT_DIR'
echo 'Launching real H1 sender...'
export UNITREE_NET_IFACE='$UNITREE_NET_IFACE'
export UNITREE_DOMAIN_ID='$UNITREE_DOMAIN_ID'
bash scripts/robot_h1/h1_4_start_sdk2_sender.sh
"

sleep 3
ros_quick_snapshot "after-sender-start"
process_snapshot "after-sender-start"
network_snapshot "after-sender-start"

echo
echo "Sender start script returned. Keeping debug terminal alive for monitoring."
echo "Expected:"
echo "  /upper_body/command_geom Subscription count: 1"
echo "  SDK2 rt/lowstate is received"
echo "  sender process is running"
echo
wait_forever "H1 DEBUG 3 SENDER"
