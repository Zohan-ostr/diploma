#!/usr/bin/env bash
set +e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/h1_dbg_common.sh"

print_header "H1 DEBUG 4: CALIBRATION"

log_note "Starting calibration stage"

ros_quick_snapshot "before-calibration"
process_snapshot "before-calibration"

echo
echo "Перед калибровкой оператор должен стоять перед камерой в T-позе."
echo "Проверка перед запуском:"
echo "  /pose/landmarks Publisher count должен быть 1"
echo "  /upper_body/start_calibration Subscription count должен быть 1"
echo
read -r -p "Для запуска калибровки введи CALIBRATE: " ANSWER

if [ "$ANSWER" != "CALIBRATE" ]; then
  echo "Calibration cancelled."
  exit 0
fi

capture_bash "40_calibration_start" "
cd '$PROJECT_DIR'
bash scripts/robot_h1/h1_5_start_calibration.sh
"

sleep 3

ros_quick_snapshot "after-calibration"
process_snapshot "after-calibration"

echo
echo "============================================================"
echo " CALIBRATION DEBUG DONE"
echo "============================================================"
echo "Логи:"
echo "  $LOG_ROOT"
echo "============================================================"

tail_debug_logs
