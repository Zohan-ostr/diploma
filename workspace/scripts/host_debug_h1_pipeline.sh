#!/usr/bin/env bash
set -eo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-h1_mujoco_sim}"
ROS_DOMAIN_ID_VALUE="${ROS_DOMAIN_ID_VALUE:-42}"
DEBUG_SECONDS="${DEBUG_SECONDS:-25}"

POSE_TOPIC="${POSE_TOPIC:-/pose/landmarks}"
CMD_TOPIC="${CMD_TOPIC:-/upper_body/command_geom}"
OLD_CMD_TOPIC="${OLD_CMD_TOPIC:-/upper_body/command}"
LOWCMD_TOPIC="${LOWCMD_TOPIC:-/lowcmd}"
LOWSTATE_TOPIC="${LOWSTATE_TOPIC:-/lowstate}"

CID="$(docker ps --filter "name=^/${CONTAINER_NAME}$" --format "{{.ID}}" | head -n 1)"

if [ -z "$CID" ]; then
  echo "ERROR: running container '$CONTAINER_NAME' not found."
  echo
  docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}"
  exit 1
fi

echo "Found container:"
docker ps --filter "id=$CID" --format "table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}"

echo
echo "Running debug inside container..."
echo "  CONTAINER_NAME: $CONTAINER_NAME"
echo "  DEBUG_SECONDS:  $DEBUG_SECONDS"
echo "  POSE_TOPIC:     $POSE_TOPIC"
echo "  CMD_TOPIC:      $CMD_TOPIC"
echo "  LOWCMD_TOPIC:   $LOWCMD_TOPIC"
echo "  LOWSTATE_TOPIC: $LOWSTATE_TOPIC"
echo

docker exec -it \
  -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID_VALUE" \
  -e ROS_LOCALHOST_ONLY=0 \
  -e DEBUG_SECONDS="$DEBUG_SECONDS" \
  -e POSE_TOPIC="$POSE_TOPIC" \
  -e CMD_TOPIC="$CMD_TOPIC" \
  -e OLD_CMD_TOPIC="$OLD_CMD_TOPIC" \
  -e LOWCMD_TOPIC="$LOWCMD_TOPIC" \
  -e LOWSTATE_TOPIC="$LOWSTATE_TOPIC" \
  "$CID" bash -lc 'cat > /tmp/debug_h1_pipeline.sh <<'"'"'INNER_BASH'"'"'
#!/usr/bin/env bash
set -eo pipefail

cd /workspace || exit 1

# ROS setup scripts may reference unset variables, so do not use set -u here.
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash 2>/dev/null || true

set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

POSE_TOPIC="${POSE_TOPIC:-/pose/landmarks}"
CMD_TOPIC="${CMD_TOPIC:-/upper_body/command_geom}"
OLD_CMD_TOPIC="${OLD_CMD_TOPIC:-/upper_body/command}"
LOWCMD_TOPIC="${LOWCMD_TOPIC:-/lowcmd}"
LOWSTATE_TOPIC="${LOWSTATE_TOPIC:-/lowstate}"
DEBUG_SECONDS="${DEBUG_SECONDS:-25}"

echo "============================================================"
echo " H1 PIPELINE DEBUG INSIDE CONTAINER"
echo " hostname: $(hostname)"
echo " ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo " ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY"
echo " DEBUG_SECONDS=$DEBUG_SECONDS"
echo "============================================================"
echo

echo "===== 1) Processes ====="
ps aux | grep -E "h1_geometric|h1_sdk2py|retarget|mediapipe|mujoco|home_camera|webcam|camera" | grep -v grep || true
echo

echo "===== 2) Topics ====="
ros2 topic list -t | grep -E "pose|upper_body|lowcmd|lowstate|arm_sdk|camera" || true
echo

echo "===== 3) Topic graph ====="
for t in "$POSE_TOPIC" "$OLD_CMD_TOPIC" "$CMD_TOPIC" "$LOWCMD_TOPIC" "$LOWSTATE_TOPIC"; do
  echo
  echo "--- $t ---"
  ros2 topic info "$t" -v 2>/dev/null || echo "NO SUCH TOPIC"
done

echo
echo "===== 4) Message frequency, 5 sec each ====="
for t in "$POSE_TOPIC" "$OLD_CMD_TOPIC" "$CMD_TOPIC" "$LOWCMD_TOPIC" "$LOWSTATE_TOPIC"; do
  echo
  echo "--- hz $t ---"
  timeout 6 ros2 topic hz "$t" 2>/dev/null || echo "NO HZ / TIMEOUT"
done

echo
echo "===== 5) One landmarks message ====="
timeout 5 ros2 topic echo "$POSE_TOPIC" --once 2>/dev/null || echo "NO $POSE_TOPIC MESSAGE"

echo
echo "===== 6) One geometric command message ====="
timeout 5 ros2 topic echo "$CMD_TOPIC" --once 2>/dev/null || echo "NO $CMD_TOPIC MESSAGE"

echo
echo "===== 7) Python live diagnostic ====="
python3 - <<PY
import os
import time
import numpy as np

import rclpy
from rclpy.node import Node

from upper_body_msgs.msg import PoseLandmarks3D, UpperBodyCommand
from unitree_go.msg import LowState, LowCmd

POSE_TOPIC = os.environ.get("POSE_TOPIC", "/pose/landmarks")
CMD_TOPIC = os.environ.get("CMD_TOPIC", "/upper_body/command_geom")
OLD_CMD_TOPIC = os.environ.get("OLD_CMD_TOPIC", "/upper_body/command")
LOWCMD_TOPIC = os.environ.get("LOWCMD_TOPIC", "/lowcmd")
LOWSTATE_TOPIC = os.environ.get("LOWSTATE_TOPIC", "/lowstate")
DEBUG_SECONDS = float(os.environ.get("DEBUG_SECONDS", "25"))

CMD_ORDER = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

class DebugNode(Node):
    def __init__(self):
        super().__init__("h1_pipeline_debug_node")

        self.t0 = time.time()

        self.counts = {
            "landmarks": 0,
            "old_cmd": 0,
            "cmd": 0,
            "lowcmd": 0,
            "lowstate": 0,
        }

        self.last_t = {k: None for k in self.counts}

        self.last_vis = {}

        self.first_cmd = None
        self.last_cmd = None

        self.first_old_cmd = None
        self.last_old_cmd = None

        self.first_lowcmd_arm = None
        self.last_lowcmd_arm = None

        self.first_lowstate_arm = None
        self.last_lowstate_arm = None

        self.create_subscription(PoseLandmarks3D, POSE_TOPIC, self.on_landmarks, 10)
        self.create_subscription(UpperBodyCommand, OLD_CMD_TOPIC, self.on_old_cmd, 10)
        self.create_subscription(UpperBodyCommand, CMD_TOPIC, self.on_cmd, 10)
        self.create_subscription(LowCmd, LOWCMD_TOPIC, self.on_lowcmd, 10)
        self.create_subscription(LowState, LOWSTATE_TOPIC, self.on_lowstate, 10)

        self.create_timer(1.0, self.report)

    def parse_upper_cmd(self, msg):
        if len(msg.position) < 8:
            return None

        if len(msg.joint_names) == len(msg.position):
            by_name = dict(zip(msg.joint_names, msg.position))
            if all(name in by_name for name in CMD_ORDER):
                return np.array([by_name[name] for name in CMD_ORDER], dtype=float)

        return np.array(msg.position[:8], dtype=float)

    def on_landmarks(self, msg):
        self.counts["landmarks"] += 1
        self.last_t["landmarks"] = time.time()

        wanted = [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
        ]

        vis = {}
        for i, name in enumerate(msg.names):
            if name in wanted:
                vis[name] = float(msg.visibility[i]) if i < len(msg.visibility) else -1.0

        self.last_vis = vis

    def on_old_cmd(self, msg):
        self.counts["old_cmd"] += 1
        self.last_t["old_cmd"] = time.time()

        q = self.parse_upper_cmd(msg)
        if q is None:
            return
        if self.first_old_cmd is None:
            self.first_old_cmd = q.copy()
        self.last_old_cmd = q.copy()

    def on_cmd(self, msg):
        self.counts["cmd"] += 1
        self.last_t["cmd"] = time.time()

        q = self.parse_upper_cmd(msg)
        if q is None:
            return
        if self.first_cmd is None:
            self.first_cmd = q.copy()
        self.last_cmd = q.copy()

    def on_lowcmd(self, msg):
        self.counts["lowcmd"] += 1
        self.last_t["lowcmd"] = time.time()

        try:
            q = np.array([
                msg.motor_cmd[16].q,
                msg.motor_cmd[17].q,
                msg.motor_cmd[18].q,
                msg.motor_cmd[19].q,
                msg.motor_cmd[12].q,
                msg.motor_cmd[13].q,
                msg.motor_cmd[14].q,
                msg.motor_cmd[15].q,
            ], dtype=float)
        except Exception:
            return

        if self.first_lowcmd_arm is None:
            self.first_lowcmd_arm = q.copy()
        self.last_lowcmd_arm = q.copy()

    def on_lowstate(self, msg):
        self.counts["lowstate"] += 1
        self.last_t["lowstate"] = time.time()

        try:
            q = np.array([
                msg.motor_state[16].q,
                msg.motor_state[17].q,
                msg.motor_state[18].q,
                msg.motor_state[19].q,
                msg.motor_state[12].q,
                msg.motor_state[13].q,
                msg.motor_state[14].q,
                msg.motor_state[15].q,
            ], dtype=float)
        except Exception:
            return

        if self.first_lowstate_arm is None:
            self.first_lowstate_arm = q.copy()
        self.last_lowstate_arm = q.copy()

    def age(self, key):
        t = self.last_t.get(key)
        if t is None:
            return "never"
        return f"{time.time() - t:.2f}s ago"

    def delta_norm(self, a, b):
        if a is None or b is None:
            return 0.0
        return float(np.linalg.norm(a - b))

    def report(self):
        elapsed = max(1e-6, time.time() - self.t0)

        print()
        print("============================================================")
        print(f"elapsed: {elapsed:.1f}s")

        for key in ["landmarks", "old_cmd", "cmd", "lowcmd", "lowstate"]:
            print(f"{key:10s}: count={self.counts[key]:5d}, hz={self.counts[key]/elapsed:6.2f}, last={self.age(key)}")

        if self.last_vis:
            print("visibility:")
            for k in ["left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist"]:
                print(f"  {k:16s}: {self.last_vis.get(k, -1.0):.3f}")

        if self.last_old_cmd is not None:
            print("last OLD /upper_body/command q:")
            print(np.array2string(self.last_old_cmd, precision=3, suppress_small=True))
            print(f"old command delta: {self.delta_norm(self.last_old_cmd, self.first_old_cmd):.4f}")

        if self.last_cmd is not None:
            print("last GEOM /upper_body/command_geom q:")
            print(np.array2string(self.last_cmd, precision=3, suppress_small=True))
            print(f"geom command delta: {self.delta_norm(self.last_cmd, self.first_cmd):.4f}")

        if self.last_lowcmd_arm is not None:
            print("last /lowcmd arm target q:")
            print(np.array2string(self.last_lowcmd_arm, precision=3, suppress_small=True))
            print(f"lowcmd delta: {self.delta_norm(self.last_lowcmd_arm, self.first_lowcmd_arm):.4f}")

        if self.last_lowstate_arm is not None:
            print("last /lowstate arm q:")
            print(np.array2string(self.last_lowstate_arm, precision=3, suppress_small=True))
            print(f"lowstate delta: {self.delta_norm(self.last_lowstate_arm, self.first_lowstate_arm):.4f}")

        print()
        print("DIAGNOSIS:")

        if self.counts["landmarks"] == 0:
            print("  ❌ Нет landmarks. В этом контейнере не запущена камера/MediaPipe или она запущена в другом контейнере/ROS_DOMAIN.")
        elif self.counts["cmd"] == 0:
            print("  ❌ Landmarks есть, но нет /upper_body/command_geom.")
            print("     Значит clean geometric retarget не запущен, слушает не тот topic или не прошёл калибровку.")
        elif self.counts["lowcmd"] == 0:
            print("  ❌ /upper_body/command_geom есть, но /lowcmd нет.")
            print("     Значит bridge не запущен или упал.")
        elif self.counts["lowstate"] == 0:
            print("  ❌ /lowcmd есть, но /lowstate нет.")
            print("     Значит MuJoCo/DDS состояние не приходит.")
        else:
            geom_delta = self.delta_norm(self.last_cmd, self.first_cmd)
            lowcmd_delta = self.delta_norm(self.last_lowcmd_arm, self.first_lowcmd_arm)
            lowstate_delta = self.delta_norm(self.last_lowstate_arm, self.first_lowstate_arm)

            if geom_delta < 0.02:
                print("  ⚠️ /upper_body/command_geom почти не меняется.")
                print("     Retarget публикует, но либо ты не двигался после калибровки, либо калибровка/видимость/геометрия плохая.")
            elif lowcmd_delta < 0.02:
                print("  ⚠️ command_geom меняется, но /lowcmd arm target почти не меняется.")
                print("     Проблема в bridge: он не применяет входные команды.")
            elif lowstate_delta < 0.02:
                print("  ⚠️ /lowcmd target меняется, но /lowstate arm почти не меняется.")
                print("     Команды не двигают MuJoCo: mode/kp/kd/DDS/конфликт publisher.")
            else:
                print("  ✅ Цепочка живая: landmarks → command_geom → lowcmd → lowstate.")
                print("     Если движения кривые, настраиваем знаки/масштабы retarget.")

        print("============================================================")

def main():
    rclpy.init()
    node = DebugNode()
    end = time.time() + DEBUG_SECONDS
    while rclpy.ok() and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
PY

INNER_BASH

chmod +x /tmp/debug_h1_pipeline.sh
bash /tmp/debug_h1_pipeline.sh'
