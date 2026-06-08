#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."
source scripts/robot_h1/h1_env.sh

TOPIC="${TOPIC:-/upper_body/start_calibration}"
COUNTDOWN="${COUNTDOWN:-3}"
RATE_HZ="${RATE_HZ:-10}"
DURATION_SEC="${DURATION_SEC:-3}"

print_h1_env

echo "============================================================"
echo " H1 5: START CALIBRATION"
echo "============================================================"
echo "TOPIC:        $TOPIC"
echo "COUNTDOWN:    $COUNTDOWN sec"
echo "PUBLISH:      $RATE_HZ Hz for $DURATION_SEC sec"
echo "============================================================"

echo
echo "Встань перед камерой в T-позу с прямыми руками."
echo "Калибровка начнётся через $COUNTDOWN секунды."
echo

for ((i=COUNTDOWN; i>=1; i--)); do
  echo "$i..."
  sleep 1
done

echo "START CALIBRATION"

ros2 topic pub "$TOPIC" std_msgs/msg/Bool "{data: true}" \
  --rate "$RATE_HZ" \
  --times "$((RATE_HZ * DURATION_SEC))"
