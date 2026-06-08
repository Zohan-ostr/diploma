#!/usr/bin/env bash
set +e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/h1_dbg_common.sh"

print_header "H1 DEBUG 5: COLLECT / FINAL CHECK"

log_note "Collecting final diagnostics"

network_snapshot "collect-start"
process_snapshot "collect-start"
ros_quick_snapshot "collect-start"

capture_bash "50_h1_check_script" "
cd '$PROJECT_DIR'
bash scripts/robot_h1/h1_6_check.sh
"

capture_bash "51_topic_samples" "
cd '$PROJECT_DIR'
echo '=== /pose/landmarks sample ==='
docker exec -i '$CAMERA_CONTAINER' bash -lc '
cd /workspace 2>/dev/null || true
source /opt/ros/humble/setup.bash 2>/dev/null || true
source install/setup.bash 2>/dev/null || true
source install/upper_body_msgs/share/upper_body_msgs/local_setup.bash 2>/dev/null || true
export ROS_DOMAIN_ID=$ROS_DOMAIN_ID_VALUE
timeout 5 ros2 topic echo /pose/landmarks --once
' || true

echo
echo '=== /upper_body/command_geom sample ==='
docker exec -i '$CAMERA_CONTAINER' bash -lc '
cd /workspace 2>/dev/null || true
source /opt/ros/humble/setup.bash 2>/dev/null || true
source install/setup.bash 2>/dev/null || true
source install/upper_body_msgs/share/upper_body_msgs/local_setup.bash 2>/dev/null || true
export ROS_DOMAIN_ID=$ROS_DOMAIN_ID_VALUE
timeout 5 ros2 topic echo /upper_body/command_geom --once
' || true
"

capture_bash "52_recent_logs" "
cd '$PROJECT_DIR'
echo '=== recent logs under logs/ ==='
find logs -type f 2>/dev/null | sort | tail -n 80

echo
echo '=== tail selected logs ==='
for f in \$(find logs -type f 2>/dev/null | sort | tail -n 40); do
  echo
  echo '--------------------' \$f '--------------------'
  tail -n 80 \"\$f\" 2>/dev/null || true
done
"

ARCHIVE="$LOG_BASE/h1_debug_${RUN_ID}.tar.gz"

capture_bash "53_archive" "
cd '$PROJECT_DIR'
tar -czf '$ARCHIVE' -C '$LOG_BASE' '$RUN_ID'
ls -lh '$ARCHIVE'
"

echo
echo "============================================================"
echo " FINAL COLLECT DONE"
echo "============================================================"
echo "Папка логов:"
echo "  $LOG_ROOT"
echo
echo "Архив:"
echo "  $ARCHIVE"
echo
echo "Можно отправить мне:"
echo "  1) вывод терминалов;"
echo "  2) содержимое logs/robot_h1_debug/latest/;"
echo "  3) архив $ARCHIVE."
echo "============================================================"

tail_debug_logs
